import torch

fp4_map = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

x = fp4_map.unsqueeze(0).repeat(4, 1)

xmin = x.min(dim=1, keepdim=True).values
xmax = x.max(dim=1, keepdim=True).values

scale = (xmax - xmin) / 3.0
scale = scale.clamp(min=1e-8)

# Force zero to be an exact integer level
zero_level = torch.round(-xmin / scale)
zero_level = zero_level.clamp(0, 3)
m0 = -zero_level * scale

x_q = torch.round((x - m0) / scale).to(torch.uint8)
x_q = x_q.clamp(0, 3)

x_recon = x_q.float() * scale + m0
print("Original x:", x[0])
print("Reconstructed:", x_recon[0])
print("Max error:", (x[0] - x_recon[0]).abs().max().item())

# Also try fixing m0 and optimizing scale with a simple 1D search
