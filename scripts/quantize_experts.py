#!/usr/bin/env python3
"""
quantize_experts.py - Convert moe_experts.bin (FP4) to 2-bit or 3-bit with optional Imatrix weighting.

Usage:
  python3 scripts/quantize_experts.py --bits 2 --manifest moecher_manifest.json
  python3 scripts/quantize_experts.py --bits 2 --manifest moecher_manifest.json --imatrix path/to/imatrix.dat
"""

import os
import json
import struct
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


class ImatrixLoader:
    """
    Loads and parses activation importance matrices (.dat binary format, .pt, or .npy).
    Standard .dat format from llama.cpp:
      int32_t n_entries
      for each entry:
        int32_t name_len
        char name[name_len]
        int32_t ncall
        int32_t nval
        float32 values[nval] (flattened [n_experts * n_cols] or [n_cols])
    """
    def __init__(self, path: str):
        self.path = path
        self.entries = {}
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Imatrix file not found: {path}")

        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu")
            self.entries = data if isinstance(data, dict) else {"data": data}
        elif path.endswith(".npy"):
            import numpy as np
            data = np.load(path, allow_pickle=True)
            self.entries = data.item() if data.dtype == object else {"data": torch.from_numpy(data)}
        else:
            self._load_dat(path)

        print(f"Loaded imatrix from {path} with {len(self.entries)} tensor entries.")

    def _load_dat(self, path: str):
        with open(path, "rb") as f:
            raw_entries = f.read(4)
            if len(raw_entries) < 4:
                raise ValueError("Corrupt imatrix header")
            n_entries = struct.unpack("<i", raw_entries)[0]
            for _ in range(n_entries):
                raw_len = f.read(4)
                if not raw_len:
                    break
                name_len = struct.unpack("<i", raw_len)[0]
                name = f.read(name_len).decode("utf-8", errors="ignore")
                ncall = struct.unpack("<i", f.read(4))[0]
                nval = struct.unpack("<i", f.read(4))[0]
                val_bytes = f.read(nval * 4)
                vals = torch.frombuffer(bytearray(val_bytes), dtype=torch.float32).clone()
                self.entries[name] = vals

    def get_expert_weights(self, layer_idx: int, expert_idx: int, part_name: str, cols: int, device=None) -> torch.Tensor:
        """
        Retrieves column importance weights for a specific layer, expert, and part.
        Returns a tensor of shape [cols] or None if not found.
        """
        candidate_names = []
        if "w1" in part_name or "gate" in part_name:
            candidate_names.extend([
                f"blk.{layer_idx}.ffn_gate_exps.weight",
                f"layers.{layer_idx}.mlp.experts.{expert_idx}.gate_proj.weight",
                f"layers.{layer_idx}.ffn_gate_exps.weight"
            ])
        elif "w3" in part_name or "up" in part_name:
            candidate_names.extend([
                f"blk.{layer_idx}.ffn_up_exps.weight",
                f"layers.{layer_idx}.mlp.experts.{expert_idx}.up_proj.weight",
                f"layers.{layer_idx}.ffn_up_exps.weight"
            ])
        elif "w2" in part_name or "down" in part_name:
            candidate_names.extend([
                f"blk.{layer_idx}.ffn_down_exps.weight",
                f"layers.{layer_idx}.mlp.experts.{expert_idx}.down_proj.weight",
                f"layers.{layer_idx}.ffn_down_exps.weight"
            ])

        for name in candidate_names:
            if name in self.entries:
                vals = self.entries[name]
                if vals.numel() >= (expert_idx + 1) * cols:
                    slice_vals = vals[expert_idx * cols : (expert_idx + 1) * cols]
                elif vals.numel() == cols:
                    slice_vals = vals
                else:
                    continue
                # Normalize importance weights so mean is 1.0
                slice_vals = slice_vals.clamp(min=1e-6)
                slice_vals = slice_vals / (slice_vals.mean() + 1e-8)
                if device is not None:
                    slice_vals = slice_vals.to(device)
                return slice_vals

        return None


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


def quantize_to_int2_asymmetric(x, block_size=64, col_weights=None):
    """
    Quantize fp32 tensor x [rows, cols] to INT2 (4 values per byte) asymmetric using MSE optimization.
    Optionally weights MSE by activation importance `col_weights` [cols] (imatrix).
    Returns: packed_bytes, scales, mins
    """
    rows, cols = x.shape
    assert cols % block_size == 0
    
    x = x.view(rows, cols // block_size, block_size)
    
    N = rows * (cols // block_size)
    x_flat = x.view(N, block_size).unsqueeze(1) # [N, 1, block_size]

    if col_weights is not None:
        w = col_weights.view(1, cols // block_size, block_size).expand(rows, -1, -1)
        w_flat = w.reshape(N, 1, block_size)
    else:
        w_flat = None

    x_sorted = torch.sort(x_flat, dim=2).values
    
    best_loss = torch.full((N,), float('inf'), device=x.device)
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
        
        sq_err = (x_flat - recon) ** 2
        if w_flat is not None:
            loss = (sq_err * w_flat).mean(dim=2) # [N, 1]
        else:
            err_sorted = torch.sort(sq_err, dim=2).values
            loss = err_sorted[:, :, :k_best].mean(dim=2) # [N, 1]
        
        min_loss = loss.squeeze(1) # [N]
        improved = min_loss < best_loss
        
        best_loss[improved] = min_loss[improved]
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
    parser = argparse.ArgumentParser(description="Quantize DeepSeek MoE experts to INT2 with optional Imatrix.")
    parser.add_argument("--format", type=str, default="int2", choices=["int2", "iq2_xxs"],
                        help="Quantization format: 'int2' (scalar uniform) or 'iq2_xxs' (vector lattice IQ2_XXS + Q2_K)")
    parser.add_argument("--bits", type=int, default=2, choices=[2, 3])
    parser.add_argument("--manifest", type=str, default="moecher_manifest.json",
                        help="Path to base manifest file (default: moecher_manifest.json)")
    parser.add_argument("--imatrix", type=str, default=None,
                        help="Optional path to importance matrix file (.dat, .pt, or .npy)")
    parser.add_argument("--output-manifest", "-o", type=str, default=None,
                        help="Path for output manifest (default: moecher_manifest_q2_imatrix.json if --imatrix else moecher_manifest_q2.json)")
    parser.add_argument("--output-bin", type=str, default=None,
                        help="Path for output expert binary (default: moe_experts_q2_imatrix.bin if --imatrix else moe_experts_q2.bin)")
    args = parser.parse_args()

    if args.format == "iq2_xxs":
        import subprocess
        cmd = [sys.executable, "scripts/quantize_experts_iq2.py", "--manifest", args.manifest]
        if args.imatrix:
            cmd.extend(["--imatrix", args.imatrix])
        if args.output_manifest:
            cmd.extend(["--output-manifest", args.output_manifest])
        if args.output_bin:
            cmd.extend(["--output-bin", args.output_bin])
        sys.exit(subprocess.call(cmd))
    
    if args.bits != 2:
        raise NotImplementedError("Only 2-bit is fully supported in this script so far.")
        
    with open(args.manifest, "r") as f:
        manifest = json.load(f)
        
    engine_cfg = manifest["model_config"]
    if engine_cfg.get("expert_dtype") != "fp4":
        print("Error: Input manifest must have expert_dtype == 'fp4'")
        return
        
    imatrix_loader = None
    if args.imatrix:
        imatrix_loader = ImatrixLoader(args.imatrix)

    expert_bin_path = manifest["expert_bin"]
    layout = manifest["expert_layout"]
    parts = layout["parts"]
    part_order = layout["part_order"]
    block_size = layout["block_size"]
    n_layers = layout["n_layers"]
    n_experts = layout["n_experts"]
    total_blocks = n_layers * n_experts
    
    # Determine output filenames safely
    tag = "_q2_imatrix" if args.imatrix else "_q2"
    if args.output_bin:
        out_bin = args.output_bin
    else:
        out_bin = str(Path(expert_bin_path).parent / f"moe_experts{tag}.bin")
        
    if args.output_manifest:
        out_manifest = args.output_manifest
    else:
        base_stem = Path(args.manifest).stem
        if base_stem.endswith("_q2") or base_stem.endswith("_imatrix"):
            base_stem = "moecher_manifest"
        out_manifest = str(Path(args.manifest).parent / f"{base_stem}{tag}.json")
    
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
    if args.imatrix:
        new_manifest["model_config"]["imatrix"] = args.imatrix
    
    new_manifest["expert_bin"] = out_bin
    
    print(f"Writing quantized weights to {out_bin}...")
    print(f"Target manifest will be saved to {out_manifest}...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    with open(expert_bin_path, "rb") as fin, open(out_bin, "wb") as fout:
        for b in tqdm(range(total_blocks)):
            layer_idx = b // n_experts
            expert_idx = b % n_experts

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
                
                col_weights = None
                if imatrix_loader:
                    col_weights = imatrix_loader.get_expert_weights(
                        layer_idx, expert_idx, w_part, logical_cols, device=device
                    )
                
                packed_int2, scale_bf16, min_bf16 = quantize_to_int2_asymmetric(
                    fp32_vals, block_size=q_block_size, col_weights=col_weights
                )
                
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
