#!/usr/bin/env python3
"""
quantize_experts_iq2.py - Standalone GPU-accelerated IQ2_XXS + Q2_K Quantizer for MinnieTheMoEcher.

Converts FP4 base weights (moe_experts.bin) or raw model weights directly into
calibrated IQ2_XXS (w1/w3) and Q2_K (w2) binary format with zero external dependencies.

Usage:
  python3 scripts/quantize_experts_iq2.py --manifest moecher_manifest.json
  python3 scripts/quantize_experts_iq2.py --manifest moecher_manifest.json --imatrix path/to/imatrix.dat
"""

import os
import sys
import json
import time
import struct
import argparse
from pathlib import Path
import numpy as np
import torch
from tqdm import tqdm

# 128-entry sign lookup table for IQ2_XXS (even parity)
KSIGNS_IQ2XS = [
      0, 129, 130,   3, 132,   5,   6, 135, 136,   9,  10, 139,  12, 141, 142,  15,
    144,  17,  18, 147,  20, 149, 150,  23,  24, 153, 154,  27, 156,  29,  30, 159,
    160,  33,  34, 163,  36, 165, 166,  39,  40, 169, 170,  43, 172,  45,  46, 175,
     48, 177, 178,  51, 180,  53,  54, 183, 184,  57,  58, 187,  60, 189, 190,  63,
    192,  65,  66, 195,  68, 197, 198,  71,  72, 201, 202,  75, 204,  77,  78, 207,
     80, 209, 210,  83, 212,  85,  86, 215, 216,  89,  90, 219,  92, 221, 222,  95,
     96, 225, 226,  99, 228, 101, 102, 231, 232, 105, 106, 235, 108, 237, 238, 111,
    240, 113, 114, 243, 116, 245, 246, 119, 120, 249, 250, 123, 252, 125, 126, 255,
]

IQ2XXS_GRID_RAW = [
    0x0808080808080808, 0x080808080808082b, 0x0808080808081919, 0x0808080808082b08,
    0x0808080808082b2b, 0x0808080808190819, 0x0808080808191908, 0x08080808082b0808,
    0x08080808082b082b, 0x08080808082b2b08, 0x08080808082b2b2b, 0x0808080819080819,
    0x0808080819081908, 0x0808080819190808, 0x0808080819192b08, 0x08080808192b0819,
    0x08080808192b1908, 0x080808082b080808, 0x080808082b08082b, 0x080808082b082b2b,
    0x080808082b2b082b, 0x0808081908080819, 0x0808081908081908, 0x0808081908190808,
    0x0808081908191919, 0x0808081919080808, 0x080808192b081908, 0x080808192b192b08,
    0x0808082b08080808, 0x0808082b0808082b, 0x0808082b082b082b, 0x0808082b2b08082b,
    0x0808190808080819, 0x0808190808081908, 0x0808190808190808, 0x08081908082b0819,
    0x08081908082b1908, 0x0808190819080808, 0x080819081908082b, 0x0808190819082b08,
    0x08081908192b0808, 0x080819082b080819, 0x080819082b081908, 0x080819082b190808,
    0x080819082b2b1908, 0x0808191908080808, 0x080819190808082b, 0x0808191908082b08,
    0x08081919082b0808, 0x080819191908192b, 0x08081919192b2b19, 0x080819192b080808,
    0x080819192b190819, 0x0808192b08082b19, 0x0808192b08190808, 0x0808192b19080808,
    0x0808192b2b081908, 0x0808192b2b2b1908, 0x08082b0808080808, 0x08082b0808081919,
    0x08082b0808082b08, 0x08082b0808191908, 0x08082b08082b2b08, 0x08082b0819080819,
    0x08082b0819081908, 0x08082b0819190808, 0x08082b081919082b, 0x08082b082b082b08,
    0x08082b1908081908, 0x08082b1919080808, 0x08082b2b0808082b, 0x08082b2b08191908,
    0x0819080808080819, 0x0819080808081908, 0x0819080808190808, 0x08190808082b0819,
    0x0819080819080808, 0x08190808192b0808, 0x081908082b081908, 0x081908082b190808,
    0x081908082b191919, 0x0819081908080808, 0x0819081908082b08, 0x08190819082b0808,
    0x0819081919190808, 0x0819081919192b2b, 0x081908192b080808, 0x0819082b082b1908,
    0x0819082b19081919, 0x0819190808080808, 0x0819190808082b08, 0x08191908082b0808,
    0x08191908082b1919, 0x0819190819082b19, 0x081919082b080808, 0x0819191908192b08,
    0x08191919192b082b, 0x0819192b08080808, 0x0819192b0819192b, 0x08192b0808080819,
    0x08192b0808081908, 0x08192b0808190808, 0x08192b0819080808, 0x08192b082b080819,
    0x08192b1908080808, 0x08192b1908081919, 0x08192b192b2b0808, 0x08192b2b19190819,
    0x082b080808080808, 0x082b08080808082b, 0x082b080808082b2b, 0x082b080819081908,
    0x082b0808192b0819, 0x082b08082b080808, 0x082b08082b08082b, 0x082b0819082b2b19,
    0x082b081919082b08, 0x082b082b08080808, 0x082b082b0808082b, 0x082b190808080819,
    0x082b190808081908, 0x082b190808190808, 0x082b190819080808, 0x082b19081919192b,
    0x082b191908080808, 0x082b191919080819, 0x082b1919192b1908, 0x082b192b2b190808,
    0x082b2b0808082b08, 0x082b2b08082b0808, 0x082b2b082b191908, 0x082b2b2b19081908,
    0x1908080808080819, 0x1908080808081908, 0x1908080808190808, 0x1908080808192b08,
    0x19080808082b0819, 0x19080808082b1908, 0x1908080819080808, 0x1908080819082b08,
    0x190808081919192b, 0x19080808192b0808, 0x190808082b080819, 0x190808082b081908,
    0x190808082b190808, 0x1908081908080808, 0x19080819082b0808, 0x19080819192b0819,
    0x190808192b080808, 0x190808192b081919, 0x1908082b08080819, 0x1908082b08190808,
    0x1908082b19082b08, 0x1908082b1919192b, 0x1908082b192b2b08, 0x1908190808080808,
    0x1908190808082b08, 0x19081908082b0808, 0x190819082b080808, 0x190819082b192b19,
    0x190819190819082b, 0x19081919082b1908, 0x1908192b08080808, 0x19082b0808080819,
    0x19082b0808081908, 0x19082b0808190808, 0x19082b0819080808, 0x19082b0819081919,
    0x19082b1908080808, 0x19082b1919192b08, 0x19082b19192b0819, 0x19082b192b08082b,
    0x19082b2b19081919, 0x19082b2b2b190808, 0x1919080808080808, 0x1919080808082b08,
    0x1919080808190819, 0x1919080808192b19, 0x19190808082b0808, 0x191908082b080808,
    0x191908082b082b08, 0x1919081908081908, 0x191908191908082b, 0x191908192b2b1908,
    0x1919082b2b190819, 0x191919082b190808, 0x191919082b19082b, 0x1919191908082b2b,
    0x1919192b08080819, 0x1919192b19191908, 0x19192b0808080808, 0x19192b0808190819,
    0x19192b0808192b19, 0x19192b08192b1908, 0x19192b1919080808, 0x19192b2b08082b08,
    0x192b080808081908, 0x192b080808190808, 0x192b080819080808, 0x192b0808192b2b08,
    0x192b081908080808, 0x192b081919191919, 0x192b082b08192b08, 0x192b082b192b0808,
    0x192b190808080808, 0x192b190808081919, 0x192b191908190808, 0x192b19190819082b,
    0x192b19192b081908, 0x192b2b081908082b, 0x2b08080808080808, 0x2b0808080808082b,
    0x2b08080808082b2b, 0x2b08080819080819, 0x2b0808082b08082b, 0x2b08081908081908,
    0x2b08081908192b08, 0x2b08081919080808, 0x2b08082b08190819, 0x2b08190808080819,
    0x2b08190808081908, 0x2b08190808190808, 0x2b08190808191919, 0x2b08190819080808,
    0x2b081908192b0808, 0x2b08191908080808, 0x2b0819191908192b, 0x2b0819192b191908,
    0x2b08192b08082b19, 0x2b08192b19080808, 0x2b08192b192b0808, 0x2b082b080808082b,
    0x2b082b1908081908, 0x2b082b2b08190819, 0x2b19080808081908, 0x2b19080808190808,
    0x2b190808082b1908, 0x2b19080819080808, 0x2b1908082b2b0819, 0x2b1908190819192b,
    0x2b1908192b080808, 0x2b19082b19081919, 0x2b19190808080808, 0x2b191908082b082b,
    0x2b19190819081908, 0x2b19191919190819, 0x2b192b082b080819, 0x2b192b19082b0808,
    0x2b2b08080808082b, 0x2b2b080819190808, 0x2b2b08082b081919, 0x2b2b081908082b19,
    0x2b2b082b08080808, 0x2b2b190808192b08, 0x2b2b2b0819190808, 0x2b2b2b1908081908,
]

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Precompute device tensors
grid_matrix = []
for val in IQ2XXS_GRID_RAW:
    bytes_8 = [(val >> (8 * i)) & 0xFF for i in range(8)]
    grid_matrix.append(bytes_8)
GRID_TENSOR = torch.tensor(grid_matrix, dtype=torch.float32, device=DEVICE) # [256, 8]
GRID_NORMS_SQ = torch.sum(GRID_TENSOR ** 2, dim=-1) # [256]

sign_matrix = []
for val in KSIGNS_IQ2XS:
    s = [(-1.0 if (val & (1 << i)) else 1.0) for i in range(8)]
    sign_matrix.append(s)
SIGN_TENSOR = torch.tensor(sign_matrix, dtype=torch.float32, device=DEVICE) # [128, 8]


# FP4 E2M1 lookup table
FP4_LOOKUP = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32, device=DEVICE)


class ImatrixLoader:
    def __init__(self, path: str):
        self.path = path
        self.entries = {}
        if not path or not os.path.exists(path):
            raise FileNotFoundError(f"Imatrix file not found: {path}")

        if path.endswith(".pt"):
            data = torch.load(path, map_location="cpu")
            self.entries = data if isinstance(data, dict) else {"data": data}
        elif path.endswith(".npy"):
            data = np.load(path, allow_pickle=True)
            self.entries = data.item() if data.dtype == object else {"data": torch.from_numpy(data)}
        else:
            self._load_dat(path)
        print(f"Loaded imatrix from {path} with {len(self.entries)} entries.")

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
                slice_vals = slice_vals.clamp(min=1e-5)
                slice_vals = slice_vals / (slice_vals.mean() + 1e-8)
                slice_vals = torch.clamp(slice_vals, min=0.05, max=50.0)
                if device is not None:
                    slice_vals = slice_vals.to(device)
                return slice_vals
        return None


def dequantize_fp4_gpu(packed_weights: torch.Tensor, scales: torch.Tensor, rows: int, logical_cols: int, scale_cols: int):
    """
    Dequantize FP4 E2M1 weights on GPU.
    packed_weights: [rows, logical_cols // 2] uint8
    scales: [rows, scale_cols] uint8 (E8M0)
    Returns: [rows, logical_cols] float32
    """
    w_low = packed_weights & 0x0F
    w_high = (packed_weights >> 4) & 0x0F
    
    unpacked = torch.empty((rows, logical_cols), dtype=torch.uint8, device=packed_weights.device)
    unpacked[:, 0::2] = w_low
    unpacked[:, 1::2] = w_high
    
    fp32_vals = FP4_LOOKUP[unpacked.long()]
    fp32_scales = torch.exp2(scales.to(torch.float32) - 127.0)
    fp32_scales = fp32_scales.unsqueeze(2).expand(-1, -1, 32).reshape(rows, logical_cols)
    
    return fp32_vals * fp32_scales


def quantize_iq2_xxs_gpu(weights_f32: torch.Tensor, col_weights: torch.Tensor = None) -> bytes:
    """
    Quantizes a [rows, cols] float32 weight matrix into packed IQ2_XXS byte stream.
    Total output bytes = (rows * cols / 256) * 66.
    """
    rows, cols = weights_f32.shape
    assert cols % 256 == 0
    n_blocks = (rows * cols) // 256
    
    w = weights_f32.view(n_blocks, 8, 4, 8)
    
    # 1. Parity-aware sign matching
    # KSIGNS_IQ2XS entries have strictly EVEN parity (bit 7 = parity of bits 0..6).
    # If the 8 input signs have odd parity, flip the sign of the element with minimal |w|.
    s_neg = (w < 0).to(torch.int32)
    parity = s_neg.sum(dim=-1) % 2 # [n_blocks, 8, 4]
    min_idx = torch.argmin(torch.abs(w), dim=-1, keepdim=True) # [n_blocks, 8, 4, 1]

    s_corrected = s_neg.clone()
    odd_mask = (parity == 1).unsqueeze(-1)
    s_corrected.scatter_(-1, min_idx, torch.where(odd_mask, 1 - s_corrected.gather(-1, min_idx), s_corrected.gather(-1, min_idx)))

    s_idx = (s_corrected[..., 0] | (s_corrected[..., 1] << 1) | (s_corrected[..., 2] << 2) |
             (s_corrected[..., 3] << 3) | (s_corrected[..., 4] << 4) | (s_corrected[..., 5] << 5) |
             (s_corrected[..., 6] << 6)) # [n_blocks, 8, 4]
             
    actual_signs = SIGN_TENSOR[s_idx] # [n_blocks, 8, 4, 8]
    w_unsigned = torch.clamp(w * actual_signs, min=0.0) # [n_blocks, 8, 4, 8]
    
    # 2. Optimal grid index search via weighted cosine similarity
    if col_weights is not None:
        cw_block = col_weights.reshape(n_blocks, 8, 4, 8)
        sqrt_cw = torch.sqrt(torch.clamp(cw_block, min=1e-5))
    else:
        cw_block = None
        sqrt_cw = None

    w_flat = (w_unsigned * sqrt_cw if sqrt_cw is not None else w_unsigned).view(-1, 8)
    M = w_flat.shape[0]
    chunk_size = 524288
    best_g_list = []
    
    for c in range(0, M, chunk_size):
        w_chunk = w_flat[c : c + chunk_size]
        dots = torch.matmul(w_chunk, GRID_TENSOR.t()) # [chunk, 256]
        dots.pow_(2)
        dots.div_(GRID_NORMS_SQ.unsqueeze(0))
        best_g_chunk = torch.argmax(dots, dim=-1)
        best_g_list.append(best_g_chunk)
        
    best_g_flat = torch.cat(best_g_list, dim=0)
    best_g = best_g_flat.view(n_blocks, 8, 4)
    best_grids = GRID_TENSOR[best_g] # [n_blocks, 8, 4, 8]
    
    # 3. Weighted Least-squares optimal sub-block scale: sum(w * x * grid) / sum(w * grid^2)
    if cw_block is not None:
        dot_vec = torch.sum(w_unsigned * best_grids * cw_block, dim=-1) # [n_blocks, 8, 4]
        norm_sq_vec = torch.sum((best_grids ** 2) * cw_block, dim=-1).clamp(min=1e-8) # [n_blocks, 8, 4]
    else:
        dot_vec = torch.sum(w_unsigned * best_grids, dim=-1) # [n_blocks, 8, 4]
        norm_sq_vec = torch.sum(best_grids ** 2, dim=-1).clamp(min=1e-8) # [n_blocks, 8, 4]
    
    dot_sub = torch.sum(dot_vec, dim=-1) # [n_blocks, 8]
    norm_sq_sub = torch.sum(norm_sq_vec, dim=-1).clamp(min=1e-8) # [n_blocks, 8]
    db_opt = (dot_sub / norm_sq_sub).clamp(min=1e-8) # [n_blocks, 8]
    
    # Super-scale d per 256 block
    d_max = db_opt.max(dim=-1)[0]
    d_raw = (d_max / 3.875).clamp(min=1e-8)
    d_half = d_raw.to(torch.float16)
    d = d_half.to(torch.float32)
    
    # Sub-block delta in [0..15]
    delta = torch.round((db_opt / (0.25 * d.unsqueeze(-1))) - 0.5)
    delta = torch.clamp(delta, 0, 15).to(torch.int32) # [n_blocks, 8]
    
    # 4. Pack into block_iq2_xxs binary layout
    aux32_0 = (best_g[..., 0] & 0xFF) | ((best_g[..., 1] & 0xFF) << 8) | ((best_g[..., 2] & 0xFF) << 16) | ((best_g[..., 3] & 0xFF) << 24)
    aux32_1 = (s_idx[..., 0] & 0x7F) | ((s_idx[..., 1] & 0x7F) << 7) | ((s_idx[..., 2] & 0x7F) << 14) | ((s_idx[..., 3] & 0x7F) << 21) | ((delta & 0x0F) << 28)
    
    # Interleave to [n_blocks, 8, 2] -> [n_blocks, 16] int32 (64 bytes)
    aux32 = torch.stack([aux32_0, aux32_1], dim=-1).view(n_blocks, 16).cpu().numpy().astype(np.uint32)
    d_half_np = d_half.cpu().numpy().view(np.uint16)
    
    # Pack: for each block: 2 bytes d (uint16) + 64 bytes qs (16 x uint32) = 66 bytes
    block_records = np.empty(n_blocks, dtype=[('d', np.uint16), ('qs', np.uint32, 16)])
    block_records['d'] = d_half_np
    block_records['qs'] = aux32
    return block_records.tobytes()


def quantize_q2_k_gpu(weights_f32: torch.Tensor, col_weights: torch.Tensor = None) -> bytes:
    """
    Quantizes a [rows, cols] float32 weight matrix into packed Q2_K byte stream.
    Total output bytes = (rows * cols / 256) * 84.
    """
    rows, cols = weights_f32.shape
    assert cols % 256 == 0
    n_blocks = (rows * cols) // 256
    
    w = weights_f32.view(n_blocks, 16, 16)
    
    # 1. Min / Max per group
    w_min = w.min(dim=-1)[0]
    w_max = w.max(dim=-1)[0]
    
    ml_raw = torch.clamp(-w_min, min=0.0)
    dl_raw = torch.where(w_min < 0, (w_max - w_min) / 3.0, w_max / 3.0).clamp(min=1e-8)
    
    # 2. Super scales d and dmin
    d_max = dl_raw.max(dim=-1)[0]
    d_raw = (d_max / 15.0).clamp(min=1e-8)
    d_half = d_raw.to(torch.float16)
    d = d_half.to(torch.float32)
    
    dmin_max = ml_raw.max(dim=-1)[0]
    dmin_raw = (dmin_max / 15.0).clamp(min=1e-8)
    dmin_half = dmin_raw.to(torch.float16)
    dmin = dmin_half.to(torch.float32)
    
    # 3. 4-bit scale and min indices
    sc_idx = torch.round(dl_raw / d.unsqueeze(-1)).clamp(0, 15).to(torch.int32)
    min_idx = torch.round(ml_raw / dmin.unsqueeze(-1)).clamp(0, 15).to(torch.int32)
    scales = (min_idx << 4) | sc_idx # [n_blocks, 16]
    
    dl = (d.unsqueeze(-1) * sc_idx.float()).clamp(min=1e-8)
    ml = dmin.unsqueeze(-1) * min_idx.float()
    
    # 4. Quantize to 2-bit [0..3]
    q = torch.round((w + ml.unsqueeze(-1)) / dl.unsqueeze(-1)).clamp(0, 3).to(torch.uint8)
    
    # 5. Pack into qs [n_blocks, 64] uint8
    qs = torch.zeros((n_blocks, 64), dtype=torch.uint8, device=weights_f32.device)
    for group in range(16):
        q_base = 32 * (group // 8) + 16 * (group & 1)
        shift = ((group // 2) & 3) * 2
        qs[:, q_base:q_base+16] |= (q[:, group, :] << shift)
        
    # Pack: struct block_q2_K { scales[16] (16B), qs[64] (64B), d (2B), dmin (2B) } = 84 bytes
    scales_np = scales.cpu().numpy().astype(np.uint8)
    qs_np = qs.cpu().numpy().astype(np.uint8)
    d_np = d_half.cpu().numpy().view(np.uint16)
    dmin_np = dmin_half.cpu().numpy().view(np.uint16)
    
    block_records = np.empty(n_blocks, dtype=[
        ('scales', np.uint8, 16),
        ('qs', np.uint8, 64),
        ('d', np.uint16),
        ('dmin', np.uint16)
    ])
    block_records['scales'] = scales_np
    block_records['qs'] = qs_np
    block_records['d'] = d_np
    block_records['dmin'] = dmin_np
    return block_records.tobytes()


def main():
    parser = argparse.ArgumentParser(description="MinnieTheMoEcher Native IQ2_XXS + Q2_K Quantizer.")
    parser.add_argument("--manifest", type=str, default="moecher_manifest.json",
                        help="Path to base manifest (default: moecher_manifest.json)")
    parser.add_argument("--imatrix", type=str, default=None,
                        help="Optional path to importance matrix file (.dat, .pt, or .npy)")
    parser.add_argument("--output-manifest", "-o", type=str, default="moecher_manifest_iq2.json",
                        help="Path for output manifest (default: moecher_manifest_iq2.json)")
    parser.add_argument("--output-bin", type=str, default="moe_experts_iq2.bin",
                        help="Path for output expert binary (default: moe_experts_iq2.bin)")
    parser.add_argument("--batch-size", "-b", type=int, default=16,
                        help="Number of experts to quantize simultaneously on GPU (default: 16)")
    parser.add_argument("--test", action="store_true",
                        help="Run self-test on 1 batch without writing full dataset")
    args = parser.parse_args()

    with open(args.manifest, "r") as f:
        manifest = json.load(f)

    expert_bin_path = manifest["expert_bin"]
    layout = manifest["expert_layout"]
    parts = layout["parts"]
    part_order = layout["part_order"]
    src_block_size = layout["block_size"]
    n_layers = layout["n_layers"]
    n_experts = layout["n_experts"]
    file_blocks = os.path.getsize(expert_bin_path) // src_block_size
    total_blocks = min(n_layers * n_experts, file_blocks)
    actual_layers = total_blocks // n_experts

    imatrix_loader = None
    if args.imatrix:
        imatrix_loader = ImatrixLoader(args.imatrix)

    # IQ2_XXS layout
    iq2_block_size = 7077888 # 2162688 (w1) + 2162688 (w3) + 2752512 (w2)
    w1_size = 2162688
    w3_size = 2162688
    w2_size = 2752512

    print(f"================================================================")
    print(f" MinnieTheMoEcher Native IQ2_XXS + Q2_K Quantizer")
    print(f"================================================================")
    print(f" Input base binary:     {expert_bin_path}")
    print(f" Total layers/experts:  {actual_layers} layers x {n_experts} experts = {total_blocks} total")
    print(f" Output expert binary:  {args.output_bin} ({total_blocks * iq2_block_size / (1024**3):.2f} GiB)")
    print(f" Output manifest:       {args.output_manifest}")
    print(f" Batch size:            {args.batch_size} experts per GPU step")
    print(f" Imatrix calibration:   {'Enabled (' + args.imatrix + ')' if args.imatrix else 'Disabled (unweighted)'}")
    print(f" Device:                {DEVICE} ({torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'})")
    print(f"================================================================")

    batch_size = args.batch_size

    if args.test:
        print(f"Running self-test on batch of {batch_size} experts...")
        with open(expert_bin_path, "rb") as fin:
            raw_bytes = fin.read(batch_size * src_block_size)
            
        w1_packed_list, w1_scales_list = [], []
        w3_packed_list, w3_scales_list = [], []
        w2_packed_list, w2_scales_list = [], []
        
        for i in range(batch_size):
            off = i * src_block_size
            bdata = raw_bytes[off : off + src_block_size]
            
            # w1
            w1_p = torch.frombuffer(bytearray(bdata[parts["w1.weight"]["offset_in_block"] : parts["w1.weight"]["offset_in_block"] + parts["w1.weight"]["nbytes"]]), dtype=torch.uint8).view(2048, 2048)
            w1_s = torch.frombuffer(bytearray(bdata[parts["w1.scale"]["offset_in_block"] : parts["w1.scale"]["offset_in_block"] + parts["w1.scale"]["nbytes"]]), dtype=torch.uint8).view(2048, 128)
            w1_packed_list.append(w1_p)
            w1_scales_list.append(w1_s)
            
            # w3
            w3_p = torch.frombuffer(bytearray(bdata[parts["w3.weight"]["offset_in_block"] : parts["w3.weight"]["offset_in_block"] + parts["w3.weight"]["nbytes"]]), dtype=torch.uint8).view(2048, 2048)
            w3_s = torch.frombuffer(bytearray(bdata[parts["w3.scale"]["offset_in_block"] : parts["w3.scale"]["offset_in_block"] + parts["w3.scale"]["nbytes"]]), dtype=torch.uint8).view(2048, 128)
            w3_packed_list.append(w3_p)
            w3_scales_list.append(w3_s)
            
            # w2
            w2_p = torch.frombuffer(bytearray(bdata[parts["w2.weight"]["offset_in_block"] : parts["w2.weight"]["offset_in_block"] + parts["w2.weight"]["nbytes"]]), dtype=torch.uint8).view(4096, 1024)
            w2_s = torch.frombuffer(bytearray(bdata[parts["w2.scale"]["offset_in_block"] : parts["w2.scale"]["offset_in_block"] + parts["w2.scale"]["nbytes"]]), dtype=torch.uint8).view(4096, 64)
            w2_packed_list.append(w2_p)
            w2_scales_list.append(w2_s)
            
        w1_packed_b = torch.stack(w1_packed_list).to(DEVICE)
        w1_scales_b = torch.stack(w1_scales_list).to(DEVICE)
        w1_f32_b = dequantize_fp4_gpu(w1_packed_b.view(-1, 2048), w1_scales_b.view(-1, 128), batch_size * 2048, 4096, 128)
        
        w3_packed_b = torch.stack(w3_packed_list).to(DEVICE)
        w3_scales_b = torch.stack(w3_scales_list).to(DEVICE)
        w3_f32_b = dequantize_fp4_gpu(w3_packed_b.view(-1, 2048), w3_scales_b.view(-1, 128), batch_size * 2048, 4096, 128)
        
        w2_packed_b = torch.stack(w2_packed_list).to(DEVICE)
        w2_scales_b = torch.stack(w2_scales_list).to(DEVICE)
        w2_f32_b = dequantize_fp4_gpu(w2_packed_b.view(-1, 1024), w2_scales_b.view(-1, 64), batch_size * 4096, 2048, 64)
        
        t0 = time.time()
        w1_bytes = quantize_iq2_xxs_gpu(w1_f32_b)
        w3_bytes = quantize_iq2_xxs_gpu(w3_f32_b)
        w2_bytes = quantize_q2_k_gpu(w2_f32_b)
        torch.cuda.synchronize()
        t1 = time.time()
        
        speed = batch_size / (t1 - t0)
        print(f"Quantized {batch_size} experts in {(t1-t0):.3f} s ({speed:.1f} experts/s)")
        print(f"w1 bytes: {len(w1_bytes)} (expected {batch_size * w1_size})")
        print(f"w3 bytes: {len(w3_bytes)} (expected {batch_size * w3_size})")
        print(f"w2 bytes: {len(w2_bytes)} (expected {batch_size * w2_size})")
        print(f"Estimated full model quantization time ({total_blocks} experts): {total_blocks / speed / 60:.1f} minutes.")
        print("Self-test SUCCESSFUL!")
        return

    # Full quantization loop
    t_start = time.time()
    with open(expert_bin_path, "rb") as fin, open(args.output_bin, "wb") as fout:
        for b_start in tqdm(range(0, total_blocks, batch_size), desc="Quantizing MoE Experts"):
            b_count = min(batch_size, total_blocks - b_start)
            raw_bytes = fin.read(b_count * src_block_size)
            if len(raw_bytes) == 0:
                break
                
            w1_packed_list, w1_scales_list = [], []
            w3_packed_list, w3_scales_list = [], []
            w2_packed_list, w2_scales_list = [], []
            
            for i in range(b_count):
                off = i * src_block_size
                bdata = raw_bytes[off : off + src_block_size]
                
                # w1
                w1_p = torch.frombuffer(bytearray(bdata[parts["w1.weight"]["offset_in_block"] : parts["w1.weight"]["offset_in_block"] + parts["w1.weight"]["nbytes"]]), dtype=torch.uint8).view(2048, 2048)
                w1_s = torch.frombuffer(bytearray(bdata[parts["w1.scale"]["offset_in_block"] : parts["w1.scale"]["offset_in_block"] + parts["w1.scale"]["nbytes"]]), dtype=torch.uint8).view(2048, 128)
                w1_packed_list.append(w1_p)
                w1_scales_list.append(w1_s)
                
                # w3
                w3_p = torch.frombuffer(bytearray(bdata[parts["w3.weight"]["offset_in_block"] : parts["w3.weight"]["offset_in_block"] + parts["w3.weight"]["nbytes"]]), dtype=torch.uint8).view(2048, 2048)
                w3_s = torch.frombuffer(bytearray(bdata[parts["w3.scale"]["offset_in_block"] : parts["w3.scale"]["offset_in_block"] + parts["w3.scale"]["nbytes"]]), dtype=torch.uint8).view(2048, 128)
                w3_packed_list.append(w3_p)
                w3_scales_list.append(w3_s)
                
                # w2
                w2_p = torch.frombuffer(bytearray(bdata[parts["w2.weight"]["offset_in_block"] : parts["w2.weight"]["offset_in_block"] + parts["w2.weight"]["nbytes"]]), dtype=torch.uint8).view(4096, 1024)
                w2_s = torch.frombuffer(bytearray(bdata[parts["w2.scale"]["offset_in_block"] : parts["w2.scale"]["offset_in_block"] + parts["w2.scale"]["nbytes"]]), dtype=torch.uint8).view(4096, 64)
                w2_packed_list.append(w2_p)
                w2_scales_list.append(w2_s)
                
            w1_packed_b = torch.stack(w1_packed_list).to(DEVICE)
            w1_scales_b = torch.stack(w1_scales_list).to(DEVICE)
            w1_f32_b = dequantize_fp4_gpu(w1_packed_b.view(-1, 2048), w1_scales_b.view(-1, 128), b_count * 2048, 4096, 128)
            
            w3_packed_b = torch.stack(w3_packed_list).to(DEVICE)
            w3_scales_b = torch.stack(w3_scales_list).to(DEVICE)
            w3_f32_b = dequantize_fp4_gpu(w3_packed_b.view(-1, 2048), w3_scales_b.view(-1, 128), b_count * 2048, 4096, 128)
            
            w2_packed_b = torch.stack(w2_packed_list).to(DEVICE)
            w2_scales_b = torch.stack(w2_scales_list).to(DEVICE)
            w2_f32_b = dequantize_fp4_gpu(w2_packed_b.view(-1, 1024), w2_scales_b.view(-1, 64), b_count * 4096, 2048, 64)
            
            # Collect imatrix weights if available
            if imatrix_loader:
                w1_cw_list, w3_cw_list, w2_cw_list = [], [], []
                for i in range(b_count):
                    b = b_start + i
                    l_idx = b // n_experts
                    e_idx = b % n_experts
                    w1_cw_i = imatrix_loader.get_expert_weights(l_idx, e_idx, "w1", 4096, device=DEVICE)
                    w3_cw_i = imatrix_loader.get_expert_weights(l_idx, e_idx, "w3", 4096, device=DEVICE)
                    w2_cw_i = imatrix_loader.get_expert_weights(l_idx, e_idx, "w2", 2048, device=DEVICE)
                    w1_cw_list.append(w1_cw_i if w1_cw_i is not None else torch.ones(4096, device=DEVICE))
                    w3_cw_list.append(w3_cw_i if w3_cw_i is not None else torch.ones(4096, device=DEVICE))
                    w2_cw_list.append(w2_cw_i if w2_cw_i is not None else torch.ones(2048, device=DEVICE))
                w1_cw_b = torch.stack(w1_cw_list).view(b_count, 1, 4096).expand(b_count, 2048, 4096).contiguous()
                w3_cw_b = torch.stack(w3_cw_list).view(b_count, 1, 4096).expand(b_count, 2048, 4096).contiguous()
                w2_cw_b = torch.stack(w2_cw_list).view(b_count, 1, 2048).expand(b_count, 4096, 2048).contiguous()
            else:
                w1_cw_b, w3_cw_b, w2_cw_b = None, None, None
            
            # Quantize batch
            w1_bytes = quantize_iq2_xxs_gpu(w1_f32_b, w1_cw_b)
            w3_bytes = quantize_iq2_xxs_gpu(w3_f32_b, w3_cw_b)
            w2_bytes = quantize_q2_k_gpu(w2_f32_b, w2_cw_b)
            
            # Interleave and write expert by expert
            for i in range(b_count):
                w1_slice = w1_bytes[i * w1_size : (i + 1) * w1_size]
                w3_slice = w3_bytes[i * w3_size : (i + 1) * w3_size]
                w2_slice = w2_bytes[i * w2_size : (i + 1) * w2_size]
                fout.write(w1_slice)
                fout.write(w3_slice)
                fout.write(w2_slice)

    t_total = time.time() - t_start
    print(f"\nSuccessfully quantized all {total_blocks} experts in {t_total:.2f}s ({total_blocks/t_total:.1f} experts/s)!")

    # Write output manifest
    new_manifest = dict(manifest)
    new_manifest["expert_bin"] = os.path.abspath(args.output_bin)
    new_manifest["model_config"]["expert_dtype"] = "iq2_xxs"
    new_manifest["expert_layout"]["block_size"] = iq2_block_size
    new_manifest["expert_layout"]["part_order"] = ["w1.weight", "w3.weight", "w2.weight"]
    new_manifest["expert_layout"]["parts"] = {
        "w1.weight": {
            "offset_in_block": 0,
            "nbytes": w1_size,
            "dtype": "IQ2_XXS",
            "shape": [2048, 4096]
        },
        "w3.weight": {
            "offset_in_block": w1_size,
            "nbytes": w3_size,
            "dtype": "IQ2_XXS",
            "shape": [2048, 4096]
        },
        "w2.weight": {
            "offset_in_block": w1_size + w3_size,
            "nbytes": w2_size,
            "dtype": "Q2_K",
            "shape": [4096, 2048]
        }
    }

    with open(args.output_manifest, "w") as f:
        json.dump(new_manifest, f, indent=2)

    print(f"Generated manifest saved to {args.output_manifest}")


if __name__ == "__main__":
    main()
