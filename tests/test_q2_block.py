import torch

torch.manual_seed(0)
x = torch.randn(100, 64)

def quantize_minmax(x):
    xmin = x.min(dim=1, keepdim=True).values
    xmax = x.max(dim=1, keepdim=True).values
    scale = (xmax - xmin) / 3.0
    x_q = torch.round((x - xmin) / scale).clamp(0, 3)
    return x_q * scale + xmin, scale, xmin

def quantize_mse(x):
    # vectorized MSE search over a grid of scales per block
    # For each block, try 64 scale candidates between 0.1*(max-min) and (max-min)
    xmin = x.min(dim=1, keepdim=True).values
    xmax = x.max(dim=1, keepdim=True).values
    rng = xmax - xmin
    scales = torch.linspace(0.1, 1.0, 64).view(1, 64, 1) * rng.unsqueeze(2)
    # try 4 zero levels
    best_recon = torch.zeros_like(x)
    best_mse = torch.ones(x.shape[0]) * float('inf')
    best_s = torch.zeros(x.shape[0], 1)
    best_m = torch.zeros(x.shape[0], 1)
    
    for zl in [0, 1, 2, 3]:
        m = -zl * scales # shape: [100, 64, 1]
        x_uns = x.unsqueeze(1) # [100, 1, 64]
        x_q = torch.round((x_uns - m) / scales).clamp(0, 3) # [100, 64, 64]
        recon = x_q * scales + m # [100, 64, 64]
        mse = ((x_uns - recon)**2).mean(dim=2) # [100, 64]
        
        min_mse, min_idx = mse.min(dim=1) # [100]
        
        # update best
        improved = min_mse < best_mse
        best_mse[improved] = min_mse[improved]
        
        best_s[improved] = scales[improved, min_idx[improved]].squeeze(-1)
        best_m[improved] = m[improved, min_idx[improved]].squeeze(-1)
        
        best_recon[improved] = recon[improved, min_idx[improved]]
        
    return best_recon, best_s, best_m

r_mm, s_mm, m_mm = quantize_minmax(x)
r_mse, s_mse, m_mse = quantize_mse(x)

print(f"Min/Max MSE: {((x - r_mm)**2).mean().item():.4f}")
print(f"MSE Opt MSE: {((x - r_mse)**2).mean().item():.4f}")
