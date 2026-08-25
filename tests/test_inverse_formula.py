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
# c=1, s=0
cos = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
sin = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]]]])

out = apply_rotary_pos_emb(q, cos, -sin)
print("c=1, s=0:", out)

# c=0, s=1
cos = torch.tensor([[[[0.0, 0.0, 0.0, 0.0]]]])
sin = torch.tensor([[[[1.0, 1.0, 1.0, 1.0]]]])
out = apply_rotary_pos_emb(q, cos, -sin)
print("c=0, s=-1:", out)

