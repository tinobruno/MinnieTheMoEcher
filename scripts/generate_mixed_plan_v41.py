#!/usr/bin/env python3
"""
generate_mixed_plan_v41.py — Generate Mixed Quantization Map for DeepSeek-V4.1-Flash
Hardware Target: 96 GB NVIDIA RTX PRO 6000 + 64 GB Host RAM
Strategy:
  - Hot Experts (Top 96 per layer = 3,840 experts = 25%): Kept in native NVFP4 (18.80 MB/expert)
  - Cold Experts (Remaining 288 per layer = 11,520 experts = 75%): Quantized to IQ2_XXS + Q2_K (9.95 MB/expert)
"""

import os
import sys
import json
import struct
import numpy as np
from pathlib import Path

def main():
    manifest_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/moecher_manifest.json")
    dense_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/attention_dense_layers.bin")
    out_map_path = Path("F:/Moecher/models/deepseek_v41_nvfp4/mixed_expert_map.bin")
    
    n_layers = 40
    n_experts = 384
    hot_per_layer = 96
    total_experts = n_layers * n_experts
    
    print("=" * 78)
    print("  DeepSeek-V4.1-Flash Mixed Quantization Planner (Approach A)")
    print(f"  Architecture: {n_layers} layers x {n_experts} experts = {total_experts} total experts")
    print(f"  Hot Allocation: {hot_per_layer} experts/layer ({hot_per_layer * n_layers} total, 25.0%) -> Native NVFP4")
    print(f"  Cold Allocation: {n_experts - hot_per_layer} experts/layer ({(n_experts - hot_per_layer) * n_layers} total, 75.0%) -> IQ2_XXS + Q2_K")
    print("=" * 78, flush=True)

    with open(manifest_path, "r") as f:
        manifest = json.load(f)
    dense_tensors = manifest.get("dense_tensors", {})

    expert_map = np.zeros(total_experts, dtype=np.uint8)

    with open(dense_path, "rb") as f_dense:
        for l in range(n_layers):
            bias_key = f"layers.{l}.ffn.gate.bias"
            if bias_key in dense_tensors:
                tinfo = dense_tensors[bias_key]
                f_dense.seek(tinfo["offset"])
                biases = np.frombuffer(f_dense.read(tinfo["nbytes"]), dtype=np.float32)
                # Rank experts descending by bias
                ranked = np.argsort(biases)[::-1]
                hot_eids = ranked[:hot_per_layer]
            else:
                # Fallback: first hot_per_layer
                hot_eids = np.arange(hot_per_layer)
                
            for eid in hot_eids:
                expert_map[l * n_experts + eid] = 1

    total_hot = int(np.sum(expert_map == 1))
    total_cold = int(np.sum(expert_map == 0))
    
    hot_size_gb = (total_hot * 18800640) / (1024**3)
    cold_size_gb = (total_cold * 9953280) / (1024**3)
    
    print(f"Hot Experts  : {total_hot:5d} ({total_hot/total_experts*100:5.1f}%) | {hot_size_gb:6.2f} GB in NVFP4 (fits in 79.5 GB VRAM)")
    print(f"Cold Experts : {total_cold:5d} ({total_cold/total_experts*100:5.1f}%) | {cold_size_gb:6.2f} GB in IQ2_XXS/Q2_K")
    print(f"Total Model MoE Storage: {hot_size_gb + cold_size_gb:.2f} GB (down from 268.95 GB)")
    
    with open(out_map_path, "wb") as f_map:
        f_map.write(expert_map.tobytes())
    print(f"\nSaved mixed expert type map to: {out_map_path}")

if __name__ == "__main__":
    main()
