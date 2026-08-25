import torch
x = torch.randn(1, 1, 256)
x[0, 0, 0] = 100.0 # outlier
x[0, 0, 1] = -100.0 # outlier

x_sorted = torch.sort(x, dim=2).values

print("Percentiles:")
for p in [0, 1, 2, 4, 8]:
    m = x_sorted[:, :, p:p+1]
    M = x_sorted[:, :, 255-p:256-p]
    scale = (M - m) / 3.0
    scale = scale.clamp(min=1e-8)
    
    x_q = torch.round((x - m) / scale).clamp(0, 3)
    recon = x_q * scale + m
    
    # Calculate MSE ignoring the outliers
    err = (x - recon)**2
    err_sorted = torch.sort(err, dim=2).values
    # mean of the 95% best reconstructed weights
    mse = err_sorted[:, :, :250].mean()
    print(f"Clip {p}: MSE of 250 best weights = {mse.item():.4f}, scale={scale.item():.4f}")

# Old method
xmin = x.min(dim=2, keepdim=True).values
xmax = x.max(dim=2, keepdim=True).values
rng = xmax - xmin
scales_grid = torch.linspace(0.1, 1.0, 16).view(1, 1, 16) * rng
scales_flat = scales_grid.view(1, 16, 1)

best_mse = float('inf')
best_scale = 0
for zl in [0, 1, 2, 3]:
    m = -zl * scales_flat
    x_q = torch.round((x - m) / scales_flat).clamp(0, 3)
    recon = x_q * scales_flat + m
    mse = ((x - recon)**2).mean()
    if mse < best_mse:
        best_mse = mse
        best_scale = scales_flat[0, 0, 0].item() # approximate

print(f"Old method full MSE: {best_mse.item():.4f}, approx scale={best_scale:.4f}")
