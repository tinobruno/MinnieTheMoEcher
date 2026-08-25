import torch

def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(x, cos, sin):
    cos = cos.repeat_interleave(2, dim=-1)
    sin = sin.repeat_interleave(2, dim=-1)
    rope_dim = cos.shape[-1]
    nope, rope = x[..., :-rope_dim], x[..., -rope_dim:]
    rotated = ((rope.float() * cos) + (rotate_half(rope).float() * sin)).to(x.dtype)
    return torch.cat([nope, rotated], dim=-1)

x = torch.arange(8).float().unsqueeze(0)
cos = (torch.arange(4).float() + 10).unsqueeze(0)
sin = (torch.arange(4).float() + 20).unsqueeze(0)
print(f"x: {x}")
print(f"cos: {cos}")
print(f"sin: {sin}")
print(f"rotated: {apply_rotary_pos_emb(x, cos, sin)}")
