import torch

fp4_map = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

x = fp4_map.unsqueeze(0).repeat(4, 1)

# We will try scales from 0.1 to 3.0
scales = torch.linspace(0.1, 4.0, 100)
best_mse = float('inf')
best_s = None
best_m = None

for s in scales:
    # Try different zero levels (0, 1, 2, 3)
    for zl in [0, 1, 2, 3]:
        m = -zl * s
        x_q = torch.round((x - m) / s)
        x_q = x_q.clamp(0, 3)
        x_recon = x_q * s + m
        mse = ((x - x_recon)**2).mean().item()
        if mse < best_mse:
            best_mse = mse
            best_s = s.item()
            best_m = m.item()

print(f"Best scale: {best_s:.4f}, Best min: {best_m:.4f}, MSE: {best_mse:.4f}")

m = best_m
s = best_s
x_q = torch.round((x - m) / s).clamp(0, 3)
x_recon = x_q * s + m
print("Reconstructed:", x_recon[0])
print("Original x:", x[0])
