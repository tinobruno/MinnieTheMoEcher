import torch

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

x = torch.arange(8).float().unsqueeze(0).unsqueeze(0) # [1, 1, 8]
cos = torch.tensor([[10.0, 20.0]]) # shape [1, 2], so rope_dim=4
sin = torch.tensor([[0.1, 0.2]])

cos = cos.repeat_interleave(2, dim=-1).unsqueeze(0)
sin = sin.repeat_interleave(2, dim=-1).unsqueeze(0)
rope_dim = cos.shape[-1]
nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
print("rope:", rope)
print("cos :", cos)
print("sin :", sin)
print("rotate_half(rope):", rotate_half(rope))

rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin))
print("rotated:", rotated)
