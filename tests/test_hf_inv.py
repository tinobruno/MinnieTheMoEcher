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
angle = torch.tensor([0.1, 0.2])
cos = torch.cat([angle.cos(), angle.cos()], dim=-1).view(1, 1, 1, 4)
sin = torch.cat([angle.sin(), angle.sin()], dim=-1).view(1, 1, 1, 4)

q_rot = apply_rotary_pos_emb(q, cos, sin)
print("q_rot:", q_rot)

# Try exact mathematical inverse:
# q_rot is interleaved. We know q_rot = q_reshaped * cos + rotate_half(q_reshaped) * sin
# q_reshaped = q_rot * cos - rotate_half(q_rot) * sin
q_reshaped = q_rot * cos - rotate_half(q_rot) * sin
print("q_reshaped:", q_reshaped)

# Then we undo the view.transpose.reshape
b, h, s, d = q_reshaped.shape
q_inv = q_reshaped.view(b, h, s, d // 2, 2).transpose(4, 3).reshape(b, h, s, d)
print("q_inv:", q_inv)

# Compare with what hf_attention.py does:
hf_inv = apply_rotary_pos_emb(q_rot, cos, -sin)
print("hf_inv:", hf_inv)

