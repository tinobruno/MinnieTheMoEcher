import torch

torch.manual_seed(0)
# Generate 6400 weights roughly like a neural network (Gaussian std=1.0)
x = torch.randn(100, 64)

best_mse = float('inf')
best_s = None
best_m = None

scales = torch.linspace(0.1, 4.0, 100)

for s in scales:
    for zl in [0, 1, 2, 3]:
        m = -zl * s
        x_q = torch.round((x - m) / s).clamp(0, 3)
        x_recon = x_q * s + m
        mse = ((x - x_recon)**2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_s = s.item()
            best_m = m.item()

print(f"Gaussian weights -> Best scale: {best_s:.4f}, Best min: {best_m:.4f}, MSE: {best_mse:.4f}")

# Compare with Min/Max
xmin = x.min(dim=1, keepdim=True).values
xmax = x.max(dim=1, keepdim=True).values
scale_mm = (xmax - xmin) / 3.0
x_q_mm = torch.round((x - xmin) / scale_mm).clamp(0, 3)
x_recon_mm = x_q_mm * scale_mm + xmin
mse_mm = ((x - x_recon_mm)**2).mean().item()

print(f"Min/Max -> MSE: {mse_mm:.4f}")
