import torch
import numpy as np
import time

device = "cuda" if torch.cuda.is_available() else "cpu"

def quantize_q2_k_torch(weights_f32: torch.Tensor):
    """
    Vectorized GPU quantization of float32 weights [N, K] into Q2_K binary bytes.
    K must be a multiple of 256.
    Returns: d_half [n_blocks], dmin_half [n_blocks], scales [n_blocks, 16] (uint8), qs [n_blocks, 64] (uint8)
    """
    N, K = weights_f32.shape
    assert K % 256 == 0
    n_blocks = (N * K) // 256
    
    # [n_blocks, 16 groups, 16 elements]
    w = weights_f32.view(n_blocks, 16, 16)
    
    # 1. Min / Max per group
    w_min = w.min(dim=-1)[0] # [n_blocks, 16]
    w_max = w.max(dim=-1)[0] # [n_blocks, 16]
    
    # ml_raw = -w_min (so that w + ml >= 0)
    ml_raw = torch.clamp(-w_min, min=0.0) # [n_blocks, 16]
    # dl_raw = (w_max - w_min) / 3.0
    dl_raw = torch.clamp((w_max - w_min) / 3.0, min=1e-8) # [n_blocks, 16]
    
    # 2. Super scales d and dmin
    d_max = dl_raw.max(dim=-1)[0] # [n_blocks]
    d_raw = (d_max / 15.0).clamp(min=1e-8)
    d_half = d_raw.to(torch.float16)
    d = d_half.to(torch.float32)
    
    dmin_max = ml_raw.max(dim=-1)[0] # [n_blocks]
    dmin_raw = (dmin_max / 15.0).clamp(min=1e-8)
    dmin_half = dmin_raw.to(torch.float16)
    dmin = dmin_half.to(torch.float32)
    
    # 3. 4-bit scale and min indices in [0..15]
    sc_idx = torch.round(dl_raw / d.unsqueeze(-1)).clamp(0, 15).to(torch.int32) # [n_blocks, 16]
    min_idx = torch.round(ml_raw / dmin.unsqueeze(-1)).clamp(0, 15).to(torch.int32) # [n_blocks, 16]
    
    # Packed scales byte: (min_idx << 4) | sc_idx
    scales = (min_idx << 4) | sc_idx # [n_blocks, 16] uint8
    
    # Reconstructed dl and ml
    dl = (d.unsqueeze(-1) * sc_idx.float()).clamp(min=1e-8) # [n_blocks, 16]
    ml = dmin.unsqueeze(-1) * min_idx.float() # [n_blocks, 16]
    
    # 4. Quantize 16 elements per group to 2 bits [0..3]
    # q = round((w + ml) / dl)
    q = torch.round((w + ml.unsqueeze(-1)) / dl.unsqueeze(-1)).clamp(0, 3).to(torch.uint8) # [n_blocks, 16, 16]
    
    # 5. Pack into qs [n_blocks, 64] uint8
    # In Q2_K:
    # q_base = 32 * (group / 8) + 16 * (group & 1)
    # shift = ((group / 2) & 3) * 2
    qs = torch.zeros((n_blocks, 64), dtype=torch.uint8, device=device)
    for group in range(16):
        q_base = 32 * (group // 8) + 16 * (group & 1)
        shift = ((group // 2) & 3) * 2
        qs[:, q_base:q_base+16] |= (q[:, group, :] << shift)
        
    return d_half, dmin_half, scales.to(torch.uint8), qs


def dequant_q2_k_torch(d_half, dmin_half, scales, qs):
    n_blocks = d_half.shape[0]
    d = d_half.to(torch.float32)
    dmin = dmin_half.to(torch.float32)
    
    recon = torch.zeros((n_blocks, 16, 16), dtype=torch.float32, device=device)
    for group in range(16):
        sc = scales[:, group].to(torch.int32)
        sc_idx = sc & 0x0F
        min_idx = sc >> 4
        dl = d * sc_idx.float()
        ml = dmin * min_idx.float()
        
        q_base = 32 * (group // 8) + 16 * (group & 1)
        shift = ((group // 2) & 3) * 2
        
        q_bytes = qs[:, q_base:q_base+16].to(torch.int32)
        q = (q_bytes >> shift) & 0x03
        
        recon[:, group, :] = dl.unsqueeze(-1) * q.float() - ml.unsqueeze(-1)
        
    return recon.view(n_blocks * 256)


# Test Q2_K
torch.manual_seed(42)
test_w = torch.randn(128, 1024, device=device, dtype=torch.float32) * 0.05

t0 = time.time()
d_half, dmin_half, scales, qs = quantize_q2_k_torch(test_w)
torch.cuda.synchronize()
t1 = time.time()

recon_w = dequant_q2_k_torch(d_half, dmin_half, scales, qs).view(128, 1024)

mse = torch.mean((test_w - recon_w) ** 2).item()
norm_w = torch.norm(test_w).item()
norm_err = torch.norm(test_w - recon_w).item()
rel_err = norm_err / norm_w

print(f"Quantized 128x1024 Q2_K weights in {(t1-t0)*1000:.2f} ms")
print(f"Relative L2 error: {rel_err:.4f} ({rel_err*100:.2f}%)")
print(f"MSE: {mse:.6e}")
