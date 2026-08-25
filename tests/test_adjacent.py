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

def apply_rope_moecher_adjacent(x, cos, sin):
    b, h, s, d = x.shape
    x = x.view(b, h, s, d//2, 2)
    x0 = x[..., 0]
    x1 = x[..., 1]
    y0 = x0 * cos - x1 * sin
    y1 = x0 * sin + x1 * cos
    return torch.stack([y0, y1], dim=-1).view(b, h, s, d)

torch.manual_seed(42)
k = torch.randn(1, 1, 1, 4)
angle = torch.tensor([0.1, 0.2])
cos = angle.cos()
sin = angle.sin()

hf_k = apply_rotary_pos_emb(k, torch.cat([cos, cos]).view(1, 1, 1, 4), torch.cat([sin, sin]).view(1, 1, 1, 4))
mo_k = apply_rope_moecher_adjacent(k, cos, sin)

print("Original k:\n", k.flatten())
print("HF k:\n", hf_k.flatten())
print("MO k:\n", mo_k.flatten())
