#!/usr/bin/env python3
"""
quantize_experts.py - Convert moe_experts.bin (FP4) to 2-bit or 3-bit.

Usage:
  python3 scripts/quantize_experts.py --bits 2 --manifest moecher_manifest.json
"""

import os
import json
import torch
import argparse
import time
from pathlib import Path
from tqdm import tqdm

# FP4 E2M1 lookup table
# Nibble 0..15 -> float
fp4_lookup = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)

def e8m0_to_float(scale_bytes):
    # scale = 2^(val - 127)
    return torch.exp2(scale_bytes.to(torch.float32) - 127.0)

def dequantize_fp4_to_fp32(packed_weights, scales, rows, logical_cols, scale_cols):
    """
    packed_weights: [rows, logical_cols // 2] uint8
    scales: [rows, scale_cols] uint8 (E8M0)
    Returns: [rows, logical_cols] float32
    """
    assert logical_cols % 2 == 0
    assert logical_cols % 32 == 0
    
    # Unpack FP4
    w_low = packed_weights & 0x0F
    w_high = (packed_weights >> 4) & 0x0F
    
    # Reshape
    w_low = w_low.view(rows, logical_cols // 2)
    w_high = w_high.view(rows, logical_cols // 2)
    
    # Interleave to [rows, logical_cols]
    unpacked = torch.empty((rows, logical_cols), dtype=torch.uint8, device=packed_weights.device)
    unpacked[:, 0::2] = w_low
    unpacked[:, 1::2] = w_high
    
    # Map from FP4 [0-15] index to real FP4 values
    fp4_map = torch.tensor([
        0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
        -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
    ], dtype=torch.float32, device=packed_weights.device)
    
    fp32_vals = fp4_map[unpacked.long()] # [rows, logical_cols]
    
    # Apply scales
    scales = scales.view(rows, scale_cols)
    fp32_scales = e8m0_to_float(scales) # [rows, scale_cols]
    fp32_scales = fp32_scales.unsqueeze(2).expand(-1, -1, 32).reshape(rows, logical_cols)
    
    return fp32_vals * fp32_scales

def quantize_to_int2_asymmetric(x, block_size=64):
    """
    Quantize fp32 tensor x [rows, cols] to INT2 (4 values per byte) asymmetric using MSE optimization.
    Returns: packed_bytes, scales, mins
    """
    rows, cols = x.shape
    assert cols % block_size == 0
    
    x = x.view(rows, cols // block_size, block_size)
    
    N = rows * (cols // block_size)
    x_flat = x.view(N, block_size).unsqueeze(1) # [N, 1, block_size]
    x_sorted = torch.sort(x_flat, dim=2).values
    
    best_mse = torch.full((N,), float('inf'), device=x.device)
    best_s = torch.zeros(N, 1, device=x.device)
    best_m = torch.zeros(N, 1, device=x.device)
    
    # We will compute MSE on the best 95% of weights to ignore outliers.
    k_best = int(block_size * 0.95)
    
    # Try different clipping percentiles (0 to 15 elements clipped from each side)
    for p in range(16):
        m = x_sorted[:, :, p:p+1]
        M = x_sorted[:, :, block_size - 1 - p:block_size - p]
        
        scale = (M - m) / 3.0
        scale = scale.clamp(min=1e-8)
        
        x_q = torch.round((x_flat - m) / scale).clamp(0, 3)
        recon = x_q * scale + m
        
        err = (x_flat - recon) ** 2
        err_sorted = torch.sort(err, dim=2).values
        mse = err_sorted[:, :, :k_best].mean(dim=2) # [N, 1]
        
        min_mse = mse.squeeze(1) # [N]
        improved = min_mse < best_mse
        
        best_mse[improved] = min_mse[improved]
        best_s[improved] = scale.squeeze(1)[improved]
        best_m[improved] = m.squeeze(1)[improved]
        
    best_s = best_s.view(rows, cols // block_size, 1)
    best_m = best_m.view(rows, cols // block_size, 1)
    
    x_q_final = torch.round((x - best_m) / best_s).clamp(0, 3).to(torch.uint8)
    x_q_final = x_q_final.view(rows, cols)
    
    scale = best_s.squeeze(-1).to(torch.bfloat16)
    xmin_final = best_m.squeeze(-1).to(torch.bfloat16)
    
    # Pack 4 values per byte
    packed = (x_q_final[:, 0::4] | 
             (x_q_final[:, 1::4] << 2) | 
             (x_q_final[:, 2::4] << 4) | 
             (x_q_final[:, 3::4] << 6))
             
    return packed, scale, xmin_final

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3])
    parser.add_argument("--manifest", type=str, default="moecher_manifest.json")
    args = parser.parse_args()
    
    if args.bits != 2:
        raise NotImplementedError("Only 2-bit is fully supported in this script so far.")
        
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
        
    engine_cfg = manifest["model_config"]
    if engine_cfg.get("expert_dtype") != "fp4":
        print("Error: Input manifest must have expert_dtype == 'fp4'")
        return
        
    expert_bin_path = manifest["expert_bin"]
    layout = manifest["expert_layout"]
    parts = layout["parts"]
    part_order = layout["part_order"]
    block_size = layout["block_size"]
    total_blocks = layout["n_layers"] * layout["n_experts"]
    
    # We will compute the new part sizes and block size for Q2
    q_block_size = 256
    new_parts = {}
    new_part_offsets = {}
    current_off = 0
    
    for part in part_order:
        orig = parts[part]
        shape = orig["shape"]
        # shape is [rows, cols_packed] for FP4 weights
        rows = shape[0]
        cols_packed = shape[1] if len(shape) > 1 else 1
        
        # We will write: packed bytes, then scales, then mins
        if orig["dtype"] == "I8": # These are the weight parts
            logical_cols = cols_packed * 2
            packed_cols = logical_cols // 4
            packed_size = rows * packed_cols
            
            new_parts[part] = {
                "offset_in_block": current_off,
                "nbytes": packed_size,
                "dtype": "INT2_PACKED",
                "shape": [rows, packed_cols]
            }
            current_off += packed_size
            
        elif orig["dtype"] == "F8_E8M0":
            prev_part = part.replace(".scale", ".weight")
            orig_weight = parts[prev_part]
            w_rows = orig_weight["shape"][0]
            w_cols_packed = orig_weight["shape"][1] if len(orig_weight["shape"]) > 1 else 1
            logical_cols = w_cols_packed * 2
            
            scale_min_size = w_rows * (logical_cols // q_block_size) * 4 # 2 bytes scale + 2 bytes min
            
            new_parts[part] = {
                "offset_in_block": current_off,
                "nbytes": scale_min_size,
                "dtype": "BF16_SCALE_MIN",
                "shape": [w_rows, logical_cols // q_block_size, 2]
            }
            current_off += scale_min_size
    
    new_expert_raw_size = current_off
    PAGE_SIZE = 4096
    new_expert_block_size = ((new_expert_raw_size + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE
    padding_bytes = new_expert_block_size - new_expert_raw_size
    
    print(f"Original expert block size: {block_size}")
    print(f"New expert block size: {new_expert_block_size}")
    
    new_manifest = dict(manifest)
    new_manifest["expert_layout"]["block_size"] = new_expert_block_size
    new_manifest["expert_layout"]["parts"] = new_parts
    new_manifest["model_config"]["expert_dtype"] = "int2"
    
    out_bin = str(Path(expert_bin_path).parent / "test_experts_q2.bin")
    new_manifest["expert_bin"] = out_bin
    
    out_manifest = str(Path(args.manifest).parent / "moecher_manifest_q2.json")
    
    print(f"Writing to {out_bin}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(expert_bin_path, "rb") as fin, open(out_bin, "wb") as fout:
        for b in tqdm(range(2)):
            fin.seek(b * block_size)
            block_data = fin.read(block_size)
            if len(block_data) == 0:
                break
                
            out_block_data = bytearray(new_expert_block_size)
            
            for i in range(0, len(part_order), 2):
                w_part = part_order[i]
                s_part = part_order[i+1]
                
                orig_w = parts[w_part]
                orig_s = parts[s_part]
                
                w_off = orig_w["offset_in_block"]
                s_off = orig_s["offset_in_block"]
                
                w_bytes = block_data[w_off : w_off + orig_w["nbytes"]]
                s_bytes = block_data[s_off : s_off + orig_s["nbytes"]]
                
                if all(b == 0 for b in w_bytes):
                    continue
                    
                rows = orig_w["shape"][0]
                cols_packed = orig_w["shape"][1] if len(orig_w["shape"]) > 1 else 1
                logical_cols = cols_packed * 2
                
                packed_fp4 = torch.frombuffer(w_bytes, dtype=torch.uint8).to(device)
                scales_fp4 = torch.frombuffer(s_bytes, dtype=torch.uint8).to(device)
                
                fp32_vals = dequantize_fp4_to_fp32(packed_fp4, scales_fp4, rows, logical_cols, logical_cols//32)
                
                packed_int2, scale_bf16, min_bf16 = quantize_to_int2_asymmetric(fp32_vals, block_size=q_block_size)
                
                n_w = new_parts[w_part]
                n_s = new_parts[s_part]
                
                out_block_data[n_w["offset_in_block"] : n_w["offset_in_block"] + n_w["nbytes"]] = packed_int2.cpu().numpy().tobytes()
                
                s_m_bytes = scale_bf16.view(torch.int16).cpu().numpy().tobytes() + min_bf16.view(torch.int16).cpu().numpy().tobytes()
                out_block_data[n_s["offset_in_block"] : n_s["offset_in_block"] + n_s["nbytes"]] = s_m_bytes
                
            fout.write(out_block_data)
            
    with open(out_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)
        
    print(f"Finished! New manifest written to {out_manifest}")

if __name__ == "__main__":
    main()
