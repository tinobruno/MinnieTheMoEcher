#!/usr/bin/env python3
"""
quantize_deepseek_sparse_nvfp4.py — Convert DeepSeek V4 Flash base FP4 experts to 2:4 Structured Sparse NVFP4.
Hardware Target: NVIDIA Blackwell (Compute Capability >= 10.0 / 12.0) & Ampere/Ada/Hopper.

Features:
  - Batched GPU-accelerated 2:4 magnitude pruning and metadata index packing
  - Block-32 E8M0 scaling with FP4 E2M1 value quantization
  - High-throughput streaming I/O with progress bar
  - Generates updated moecher_manifest.json with "sparse_nvfp4" expert layout
"""

import os
import sys
import json
import time
import shutil
import struct
import argparse
from pathlib import Path
from typing import Dict, Any, List

import torch
import numpy as np

# Midpoints between consecutive positive FP4 E2M1 values:
# [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
FP4_MIDPOINTS = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=torch.float32)
FP4_VALUES = torch.tensor([
    0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0,
    -0.0, -0.5, -1.0, -1.5, -2.0, -3.0, -4.0, -6.0
], dtype=torch.float32)


def dequantize_fp4_batch(packed_w: torch.Tensor, scales_e8m0: torch.Tensor, rows: int, cols: int) -> torch.Tensor:
    """
    Dequantizes packed FP4 [rows, cols // 2] uint8 with block-32 E8M0 scales [rows, cols // 32]
    to float32 [rows, cols].
    """
    device = packed_w.device
    fp4_vals = FP4_VALUES.to(device)
    
    w_low = packed_w & 0x0F
    w_high = (packed_w >> 4) & 0x0F
    
    unpacked = torch.empty((rows, cols), dtype=torch.uint8, device=device)
    unpacked[:, 0::2] = w_low
    unpacked[:, 1::2] = w_high
    
    f_vals = fp4_vals[unpacked.long()]
    
    scale_cols = cols // 32
    f_scales = torch.exp2(scales_e8m0.view(rows, scale_cols).to(torch.float32) - 127.0)
    f_scales_expanded = f_scales.unsqueeze(2).expand(-1, -1, 32).reshape(rows, cols)
    
    return f_vals * f_scales_expanded


def prune_and_pack_2_to_4_nvfp4(
    weight_f32: torch.Tensor,
    midpoints_gpu: torch.Tensor,
    block_size: int = 32
) -> tuple:
    """
    Applies 2:4 structured pruning along columns of weight_f32 [rows, cols],
    encodes 2:4 metadata nibbles, and quantizes retained values to NVFP4 E2M1.
    """
    rows, cols = weight_f32.shape
    num_chunks = cols // 4
    w_4 = weight_f32.view(rows, num_chunks, 4)
    w_abs = torch.abs(w_4)

    # Top-2 indices and values per chunk of 4
    top2_vals, top2_idx = torch.topk(w_abs, k=2, dim=-1, largest=True, sorted=False)
    # Sort indices in ascending order (0 <= idx0 < idx1 <= 3)
    top2_idx_sorted, _ = torch.sort(top2_idx, dim=-1)

    kept_values = torch.gather(w_4, dim=-1, index=top2_idx_sorted)
    kept_values_2d = kept_values.view(rows, cols // 2)

    # Encode 2:4 Metadata: nibble = (idx1 << 2) | idx0
    idx0 = top2_idx_sorted[:, :, 0]
    idx1 = top2_idx_sorted[:, :, 1]
    meta_nibbles = (idx1 << 2) | idx0

    # Pack 2 nibbles per byte
    meta_even = meta_nibbles[:, 0::2]
    meta_odd = meta_nibbles[:, 1::2]
    packed_meta = (meta_even | (meta_odd << 4)).to(torch.uint8)

    # NVFP4 Quantization on Kept Values
    kept_block_size = block_size // 2
    num_blocks = (cols // 2) // kept_block_size
    b = kept_values_2d.view(rows, num_blocks, kept_block_size)
    max_abs = torch.max(torch.abs(b), dim=-1, keepdim=True).values
    safe_max = torch.clamp(max_abs, min=1e-12)

    # E8M0 scale: 2^(E - 127) >= max_abs / 6.0
    exp = torch.clamp(torch.ceil(torch.log2(safe_max / 6.0)) + 127.0, 1.0, 254.0).to(torch.uint8)
    scale = torch.pow(2.0, exp.to(torch.float32) - 127.0)

    # Normalize values into [-6.0, 6.0]
    norm = b / scale
    sign = (norm < 0).to(torch.uint8)
    abs_norm = torch.abs(norm)

    fp4_idx = torch.bucketize(abs_norm, midpoints_gpu).to(torch.uint8)
    fp4_nibbles = (fp4_idx | (sign << 3)).view(rows, cols // 2)

    # Pack 2 kept FP4 values per byte
    w_even = fp4_nibbles[:, 0::2]
    w_odd = fp4_nibbles[:, 1::2]
    packed_weights = (w_even | (w_odd << 4)).contiguous()

    return packed_weights, packed_meta, exp.squeeze(-1).contiguous()


def convert_experts_sparse_nvfp4(
    input_bin: Path,
    output_bin: Path,
    manifest_in: Path,
    manifest_out: Path,
    max_layers: int = 0,
    dry_run: bool = False
):
    print("═" * 78)
    print("  DeepSeek V4 Flash — 2:4 Structured Sparse NVFP4 Expert Quantizer")
    print(f"  Input Binary       : {input_bin}")
    print(f"  Output Binary      : {output_bin}")
    print(f"  Base Manifest      : {manifest_in}")
    print(f"  Output Manifest    : {manifest_out}")
    print("═" * 78)

    with open(manifest_in, "r") as f:
        manifest = json.load(f)

    base_layout = manifest.get("expert_layout", {})
    orig_block_size = base_layout.get("block_size", 13369344)
    n_layers = base_layout.get("n_layers", 46)
    n_experts = base_layout.get("n_experts", 256)

    if max_layers > 0 and max_layers < n_layers:
        n_layers = max_layers
        print(f"Limiting conversion to first {n_layers} layers.")

    total_experts = n_layers * n_experts
    print(f"Processing {total_experts} experts ({n_layers} layers x {n_experts} experts)...")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Compute Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    midpoints_gpu = FP4_MIDPOINTS.to(device)

    # In dense FP4, offsets inside orig_block_size (13,369,344 bytes):
    # w1.weight: offset 0, [2048, 2048] bytes (logical [2048, 4096])
    # w1.scale: offset 4194304, [2048, 128] bytes
    # w3.weight: offset 4456448, [2048, 2048] bytes (logical [2048, 4096])
    # w3.scale: offset 8650752, [2048, 128] bytes
    # w2.weight: offset 8912896, [4096, 1024] bytes (logical [4096, 2048])
    # w2.scale: offset 13107200, [4096, 64] bytes
    w1_w_off, w1_w_len = 0, 4194304
    w1_s_off, w1_s_len = 4194304, 262144
    w3_w_off, w3_w_len = 4456448, 4194304
    w3_s_off, w3_s_len = 8650752, 262144
    w2_w_off, w2_w_len = 8912896, 4194304
    w2_s_off, w2_s_len = 13107200, 262144

    # Target 2:4 sparse offsets inside new block_size (10,223,616 bytes):
    # w1.weight: 2097152 bytes (offset 0)
    # w1.meta:   1048576 bytes (offset 2097152)
    # w1.scale:  262144 bytes  (offset 3145728)
    # w3.weight: 2097152 bytes (offset 3407872)
    # w3.meta:   1048576 bytes (offset 5505024)
    # w3.scale:  262144 bytes  (offset 6553600)
    # w2.weight: 2097152 bytes (offset 6815744)
    # w2.meta:   1048576 bytes (offset 8912896)
    # w2.scale:  262144 bytes  (offset 9961472)
    sparse_block_size = 10223616

    new_parts = {
        "w1.weight": {"offset_in_block": 0, "nbytes": 2097152, "dtype": "sparse_nvfp4", "shape": [2048, 1024]},
        "w1.meta":   {"offset_in_block": 2097152, "nbytes": 1048576, "dtype": "I8", "shape": [2048, 512]},
        "w1.scale":  {"offset_in_block": 3145728, "nbytes": 262144, "dtype": "F8_E8M0", "shape": [2048, 128]},
        "w3.weight": {"offset_in_block": 3407872, "nbytes": 2097152, "dtype": "sparse_nvfp4", "shape": [2048, 1024]},
        "w3.meta":   {"offset_in_block": 5505024, "nbytes": 1048576, "dtype": "I8", "shape": [2048, 512]},
        "w3.scale":  {"offset_in_block": 6553600, "nbytes": 262144, "dtype": "F8_E8M0", "shape": [2048, 128]},
        "w2.weight": {"offset_in_block": 6815744, "nbytes": 2097152, "dtype": "sparse_nvfp4", "shape": [4096, 512]},
        "w2.meta":   {"offset_in_block": 8912896, "nbytes": 1048576, "dtype": "I8", "shape": [4096, 256]},
        "w2.scale":  {"offset_in_block": 9961472, "nbytes": 262144, "dtype": "F8_E8M0", "shape": [4096, 64]}
    }

    total_dense_gb = total_experts * orig_block_size / (1024**3)
    total_sparse_gb = total_experts * sparse_block_size / (1024**3)
    print(f"Dense Input Size  : {total_dense_gb:.2f} GB")
    print(f"Sparse Output Size : {total_sparse_gb:.2f} GB (Net Savings: {total_dense_gb - total_sparse_gb:.2f} GB)")

    output_bin.parent.mkdir(parents=True, exist_ok=True)
    out_file = open(output_bin, "wb") if not dry_run else None
    in_file = open(input_bin, "rb")

    t0 = time.time()
    last_print = t0

    try:
        for exp_idx in range(total_experts):
            in_file.seek(exp_idx * orig_block_size)
            raw_block = in_file.read(orig_block_size)
            if len(raw_block) < orig_block_size:
                print(f"Warning: EOF reached at expert {exp_idx}")
                break

            # 1. w1 (gate)
            raw_w1_w = raw_block[w1_w_off : w1_w_off + w1_w_len]
            raw_w1_s = raw_block[w1_s_off : w1_s_off + w1_s_len]
            t_w1_w = torch.frombuffer(bytearray(raw_w1_w), dtype=torch.uint8).reshape(2048, 2048).to(device)
            t_w1_s = torch.frombuffer(bytearray(raw_w1_s), dtype=torch.uint8).reshape(2048, 128).to(device)
            f_w1 = dequantize_fp4_batch(t_w1_w, t_w1_s, 2048, 4096)
            sp_w1_w, sp_w1_m, sp_w1_s = prune_and_pack_2_to_4_nvfp4(f_w1, midpoints_gpu)

            # 2. w3 (up)
            raw_w3_w = raw_block[w3_w_off : w3_w_off + w3_w_len]
            raw_w3_s = raw_block[w3_s_off : w3_s_off + w3_s_len]
            t_w3_w = torch.frombuffer(bytearray(raw_w3_w), dtype=torch.uint8).reshape(2048, 2048).to(device)
            t_w3_s = torch.frombuffer(bytearray(raw_w3_s), dtype=torch.uint8).reshape(2048, 128).to(device)
            f_w3 = dequantize_fp4_batch(t_w3_w, t_w3_s, 2048, 4096)
            sp_w3_w, sp_w3_m, sp_w3_s = prune_and_pack_2_to_4_nvfp4(f_w3, midpoints_gpu)

            # 3. w2 (down)
            raw_w2_w = raw_block[w2_w_off : w2_w_off + w2_w_len]
            raw_w2_s = raw_block[w2_s_off : w2_s_off + w2_s_len]
            t_w2_w = torch.frombuffer(bytearray(raw_w2_w), dtype=torch.uint8).reshape(4096, 1024).to(device)
            t_w2_s = torch.frombuffer(bytearray(raw_w2_s), dtype=torch.uint8).reshape(4096, 64).to(device)
            f_w2 = dequantize_fp4_batch(t_w2_w, t_w2_s, 4096, 2048)
            sp_w2_w, sp_w2_m, sp_w2_s = prune_and_pack_2_to_4_nvfp4(f_w2, midpoints_gpu)

            if out_file:
                # Write in target order
                out_file.write(sp_w1_w.cpu().numpy().tobytes())
                out_file.write(sp_w1_m.cpu().numpy().tobytes())
                out_file.write(sp_w1_s.cpu().numpy().tobytes())
                out_file.write(sp_w3_w.cpu().numpy().tobytes())
                out_file.write(sp_w3_m.cpu().numpy().tobytes())
                out_file.write(sp_w3_s.cpu().numpy().tobytes())
                out_file.write(sp_w2_w.cpu().numpy().tobytes())
                out_file.write(sp_w2_m.cpu().numpy().tobytes())
                out_file.write(sp_w2_s.cpu().numpy().tobytes())

            now = time.time()
            if now - last_print > 5.0 or exp_idx == total_experts - 1:
                elapsed = now - t0
                speed = (exp_idx + 1) / elapsed
                remaining = (total_experts - (exp_idx + 1)) / (speed + 1e-6)
                layer_cur = exp_idx // n_experts
                expert_cur = exp_idx % n_experts
                print(f"[{exp_idx+1:5d}/{total_experts}] Layer {layer_cur:2d}, Expert {expert_cur:3d} | "
                      f"{speed:5.1f} experts/s | Elapsed: {elapsed/60.0:4.1f}m | ETA: {remaining/60.0:4.1f}m")
                last_print = now

    finally:
        in_file.close()
        if out_file:
            out_file.close()

    total_time = time.time() - t0
    print("\n" + "═" * 78)
    print(f"  Expert Conversion Complete in {total_time/60.0:.2f} min!")
    print(f"  Output Binary: {output_bin}")
    print("═" * 78)

    # Update manifest
    manifest["model_config"]["expert_dtype"] = "sparse_nvfp4"
    manifest["expert_bin"] = str(output_bin)
    manifest["expert_layout"] = {
        "block_size": sparse_block_size,
        "n_layers": n_layers,
        "n_experts": n_experts,
        "parts": new_parts,
        "part_order": [
            "w1.weight", "w1.meta", "w1.scale",
            "w3.weight", "w3.meta", "w3.scale",
            "w2.weight", "w2.meta", "w2.scale"
        ],
        "layer_prefixes": base_layout.get("layer_prefixes", [])[:n_layers]
    }

    if not dry_run:
        manifest_out.parent.mkdir(parents=True, exist_ok=True)
        with open(manifest_out, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote updated manifest: {manifest_out}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="DeepSeek V4 2:4 Structured Sparse NVFP4 Expert Quantizer")
    parser.add_argument("--input-bin", default="/home/tinobruno/minniethemoecher/moe_experts.bin", help="Path to base FP4 moe_experts.bin")
    parser.add_argument("--output-bin", default="models/deepseek_v4_flash_q4/moe_experts_sparse_nvfp4.bin", help="Path to output 2:4 sparse binary")
    parser.add_argument("--manifest-in", default="moecher_manifest.json", help="Path to base manifest")
    parser.add_argument("--manifest-out", default="models/deepseek_v4_flash_q4/moecher_manifest_sparse_nvfp4.json", help="Path to output manifest")
    parser.add_argument("--layers", type=int, default=0, help="Maximum number of layers to convert (0 = all)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing files")

    args = parser.parse_args()
    convert_experts_sparse_nvfp4(
        input_bin=Path(args.input_bin),
        output_bin=Path(args.output_bin),
        manifest_in=Path(args.manifest_in),
        manifest_out=Path(args.manifest_out),
        max_layers=args.layers,
        dry_run=args.dry_run
    )
