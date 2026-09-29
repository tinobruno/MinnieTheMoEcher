#!/usr/bin/env python3
"""
generate_mixed_expert_plan.py — Generate Mixed Quantization Map (Hot: NVFP4, Cold: IQ2_XXS)
Hardware Target: 96 GB NVIDIA RTX PRO 6000 Blackwell Workstation GPU
"""

import os
import sys
import json
import struct
import numpy as np
from pathlib import Path

def generate_mixed_plan(
    imatrix_path: str = "./imatrix/DeepSeek-V4-Flash-chat-v2-routed-moe-ds4-1p5m.dat",
    base_manifest_path: str = "models/deepseek_v4_flash_q4/moecher_manifest.json",
    output_manifest_path: str = "models/deepseek_v4_flash_q4/moecher_manifest_mixed.json",
    output_map_path: str = "models/deepseek_v4_flash_q4/mixed_expert_map.bin",
    hot_experts_per_layer: int = 24,
    n_layers: int = 43,
    n_experts: int = 256
):
    print("═" * 78)
    print("  DeepSeek V4 Flash — Mixed Quantization Planner (100% VRAM Resident)")
    print(f"  Hot Format : 2:4 Structured Sparse NVFP4 (10,223,616 bytes/expert)")
    print(f"  Cold Format: IQ2_XXS + Q2_K (7,077,888 bytes/expert)")
    print(f"  Target Hot Experts/Layer: {hot_experts_per_layer} / {n_experts}")
    print("═" * 78)

    layer_energies = {}
    if os.path.exists(imatrix_path):
        print(f"Loading activation importance from: {imatrix_path}")
        with open(imatrix_path, "rb") as f:
            n_entries = struct.unpack("<i", f.read(4))[0]
            for _ in range(n_entries):
                raw_len = f.read(4)
                if not raw_len: break
                name_len = struct.unpack("<i", raw_len)[0]
                name = f.read(name_len).decode("utf-8")
                ncall = struct.unpack("<i", f.read(4))[0]
                nval = struct.unpack("<i", f.read(4))[0]
                val_bytes = f.read(nval * 4)
                if "ffn_down_exps" in name:
                    l_idx = int(name.split(".")[1])
                    vals = np.frombuffer(val_bytes, dtype=np.float32).reshape(256, -1)
                    layer_energies[l_idx] = np.linalg.norm(vals, axis=1)
        print(f"Extracted importance for {len(layer_energies)} MoE layers.")
    else:
        print(f"Warning: Imatrix not found at {imatrix_path}, using uniform prior.")

    # expert_map: 0 = cold (IQ2_XXS), 1 = hot (sparse_nvfp4)
    total_experts = n_layers * n_experts
    expert_map = np.zeros(total_experts, dtype=np.uint8)

    total_hot = 0
    total_hot_energy = 0.0
    total_all_energy = 0.0

    hot_ids_by_layer = {}

    for l in range(n_layers):
        if l in layer_energies:
            scores = layer_energies[l]
            # Rank experts descending
            ranked = np.argsort(scores)[::-1]
            hot_eids = ranked[:hot_experts_per_layer].tolist()
            layer_all_e = scores.sum()
            layer_hot_e = scores[hot_eids].sum()
            total_all_energy += layer_all_e
            total_hot_energy += layer_hot_e
        else:
            # Fallback: first N experts
            hot_eids = list(range(hot_experts_per_layer))

        hot_ids_by_layer[l] = hot_eids
        for eid in hot_eids:
            expert_map[l * n_experts + eid] = 1
            total_hot += 1

    cold_experts_count = total_experts - total_hot
    hot_size_bytes = total_hot * 10223616
    cold_size_bytes = cold_experts_count * 7077888
    total_expert_bytes = hot_size_bytes + cold_size_bytes
    total_expert_gb = total_expert_bytes / (1024**3)

    print("\n" + "─" * 78)
    print(f"  Total Experts       : {total_experts} (43 layers x 256)")
    print(f"  Hot Experts (NVFP4) : {total_hot} ({total_hot/total_experts*100:.1f}%) | {hot_size_bytes/(1024**3):.2f} GB")
    print(f"  Cold Experts (IQ2)  : {cold_experts_count} ({cold_experts_count/total_experts*100:.1f}%) | {cold_size_bytes/(1024**3):.2f} GB")
    print(f"  Total MoE Footprint : {total_expert_gb:.2f} GB (100% VRAM Resident!)")
    if total_all_energy > 0:
        print(f"  Activation Energy   : {total_hot_energy/total_all_energy*100:.1f}% covered by hot NVFP4 Tensor Cores!")
    print("─" * 78 + "\n")

    # Write binary map
    Path(output_map_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_map_path, "wb") as f:
        f.write(expert_map.tobytes())
    print(f"Saved expert type map to: {output_map_path}")

    # Build manifest
    with open(base_manifest_path, "r") as f:
        manifest = json.load(f)

    manifest["model_config"]["expert_dtype"] = "mixed_nvfp4_iq2"
    manifest["expert_bins"] = {
        "cold": {
            "path": "models/deepseek_v4_flash_q4/moe_experts_iq2.bin",
            "dtype": "iq2_xxs",
            "block_size": 7077888
        },
        "hot": {
            "path": "models/deepseek_v4_flash_q4/moe_experts_sparse_nvfp4.bin",
            "dtype": "sparse_nvfp4",
            "block_size": 10223616
        }
    }
    manifest["expert_map_file"] = str(Path(output_map_path).resolve())
    manifest["expert_bin"] = "models/deepseek_v4_flash_q4/moe_experts_iq2.bin" # fallback

    with open(output_manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"Saved mixed manifest to: {output_manifest_path}")

if __name__ == "__main__":
    generate_mixed_plan()
