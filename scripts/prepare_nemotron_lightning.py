#!/usr/bin/env python3
"""
prepare_nemotron_lightning.py — Fast Zero-Copy Converter for NVIDIA Nemotron 3.5 Lightning (30B-A3B) NVFP4
Produces:
  1. attention_dense_layers.bin  — All resident weights (Mamba SSM, Attention, Router, Shared Experts)
  2. expert_weights.bin          — Page-aligned (4096-byte) routed MoE experts for O_DIRECT / Cache
  3. moecher_manifest.json       — Manifest with FrankensTin Vision Delegation

Target footprint: <14.8 GB for 100% in-VRAM resident & streaming on NVIDIA RTX 5060 Ti 16GB.
"""

import os
import sys
import json
import time
import struct
import re
from pathlib import Path

PAGE_SIZE = 4096  # Direct I/O / O_DIRECT requirement

EXPERT_PARTS = [
    ("up_proj.weight", 0, 2494464, "fp4", [1856, 1344]),
    ("up_proj.weight_scale", 2494464, 311808, "F8_E4M3", [1856, 168]),
    ("up_proj.weight_scale_2", 2806272, 4, "F32", []),
    ("down_proj.weight", 2806288, 2494464, "fp4", [2688, 928]),
    ("down_proj.weight_scale", 5300752, 311808, "F8_E4M3", [2688, 116]),
    ("down_proj.weight_scale_2", 5612560, 4, "F32", []),
]

EXPERT_RAW_SIZE = 5612564
EXPERT_BLOCK_SIZE = ((EXPERT_RAW_SIZE + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE  # 5,615,616 bytes

def read_safetensor_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header, 8 + hlen

def main():
    raw_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/dev/nemotron_raw")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "models/nemotron_3.5_lightning")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  Nemotron 3.5 Lightning (30B-A3B) FrankensTin Converter")
    print(f"  Source: {raw_dir}")
    print(f"  Target: {out_dir}")
    print("=================================================================\n")

    config_path = raw_dir / "config.json"
    index_path = raw_dir / "model.safetensors.index.json"

    if not config_path.exists() or not index_path.exists():
        print(f"Error: Missing config or index in {raw_dir}")
        sys.exit(1)

    with open(config_path, "r", encoding="utf-8") as f:
        config = json.load(f)

    with open(index_path, "r", encoding="utf-8") as f:
        index_data = json.load(f)

    weight_map = index_data.get("weight_map", {})
    print(f"Found {len(weight_map)} tensors in model index.")

    layers_block_type = config.get("layers_block_type", [])
    num_layers = len(layers_block_type)
    moe_layers = [l for l, t in enumerate(layers_block_type) if t == "moe"]
    n_routed_experts = config.get("n_routed_experts", 128)
    print(f"Total layers: {num_layers} ({len(moe_layers)} MoE layers, {sum(1 for t in layers_block_type if t == 'mamba')} Mamba, {sum(1 for t in layers_block_type if t == 'attention')} Attention)")
    print(f"Total routed experts: {len(moe_layers)} * {n_routed_experts} = {len(moe_layers) * n_routed_experts}")
    print(f"Expert block size: {EXPERT_BLOCK_SIZE} bytes (aligned to {PAGE_SIZE})")

    moe_layer_to_idx = {l: i for i, l in enumerate(moe_layers)}

    dense_shard_tensors = {}  # shard -> list of tensor_names
    expert_shard_tensors = {} # shard -> list of (tensor_name, moe_idx, expert_id, part_name)

    expert_pattern = re.compile(r"backbone\.layers\.(\d+)\.mixer\.experts\.(\d+)\.(.+)")

    for name, shard in weight_map.items():
        m = expert_pattern.match(name)
        if m:
            layer_id = int(m.group(1))
            expert_id = int(m.group(2))
            part_suffix = m.group(3)
            if layer_id in moe_layer_to_idx:
                moe_idx = moe_layer_to_idx[layer_id]
                expert_shard_tensors.setdefault(shard, []).append((name, moe_idx, expert_id, part_suffix))
        else:
            dense_shard_tensors.setdefault(shard, []).append(name)

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 1: Write attention_dense_layers.bin
    # ──────────────────────────────────────────────────────────────────────────
    dense_bin_path = out_dir / "attention_dense_layers.bin"
    print(f"\n[1/3] Writing {dense_bin_path}...")
    t0 = time.time()

    dense_manifest = {}
    current_dense_offset = 0

    with open(dense_bin_path, "wb") as dense_f:
        for shard in sorted(dense_shard_tensors.keys()):
            shard_path = raw_dir / shard
            if not shard_path.exists():
                print(f"Error: Shard {shard_path} not found!")
                sys.exit(1)

            header, data_start = read_safetensor_header(shard_path)
            with open(shard_path, "rb") as sf:
                for tensor_name in sorted(dense_shard_tensors[shard]):
                    tmeta = header[tensor_name]
                    offsets = tmeta["data_offsets"]
                    nbytes = offsets[1] - offsets[0]

                    sf.seek(data_start + offsets[0])
                    raw_bytes = sf.read(nbytes)
                    dense_f.write(raw_bytes)

                    dense_manifest[tensor_name] = {
                        "offset": current_dense_offset,
                        "nbytes": nbytes,
                        "dtype": tmeta.get("dtype", "BF16"),
                        "shape": tmeta.get("shape", [])
                    }
                    current_dense_offset += nbytes

    dense_gb = current_dense_offset / (1024 ** 3)
    print(f"  [OK] Done! Dense bin size: {dense_gb:.2f} GB in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 2: Write expert_weights.bin
    # ──────────────────────────────────────────────────────────────────────────
    expert_bin_path = out_dir / "expert_weights.bin"
    total_experts = len(moe_layers) * n_routed_experts
    total_expert_bytes = total_experts * EXPERT_BLOCK_SIZE
    expert_gb = total_expert_bytes / (1024 ** 3)
    print(f"\n[2/3] Writing {expert_bin_path} ({expert_gb:.2f} GB, {total_experts} experts)...")
    t0 = time.time()

    part_offsets = {p[0]: (p[1], p[2]) for p in EXPERT_PARTS}

    # Pre-allocate expert file
    with open(expert_bin_path, "wb") as exp_f:
        exp_f.seek(total_expert_bytes - 1)
        exp_f.write(b"\x00")

    written_count = 0
    with open(expert_bin_path, "r+b") as exp_f:
        for shard in sorted(expert_shard_tensors.keys()):
            shard_path = raw_dir / shard
            header, data_start = read_safetensor_header(shard_path)
            with open(shard_path, "rb") as sf:
                for name, moe_idx, expert_id, part_suffix in expert_shard_tensors[shard]:
                    if part_suffix not in part_offsets:
                        continue
                    part_off, part_len = part_offsets[part_suffix]
                    tmeta = header[name]
                    offsets = tmeta["data_offsets"]
                    nbytes = offsets[1] - offsets[0]

                    sf.seek(data_start + offsets[0])
                    raw_bytes = sf.read(nbytes)

                    block_idx = moe_idx * n_routed_experts + expert_id
                    file_pos = block_idx * EXPERT_BLOCK_SIZE + part_off
                    exp_f.seek(file_pos)
                    exp_f.write(raw_bytes)
                    written_count += 1

            if written_count % 1000 == 0 or shard == sorted(expert_shard_tensors.keys())[-1]:
                print(f"  Processed {shard} ({written_count}/{len(EXPERT_PARTS) * total_experts} expert parts placed)")

    print(f"  [OK] Done! Expert bin written in {time.time() - t0:.1f}s")

    # ──────────────────────────────────────────────────────────────────────────
    # PASS 3: Generate moecher_manifest.json
    # ──────────────────────────────────────────────────────────────────────────
    manifest_path = out_dir / "moecher_manifest.json"
    print(f"\n[3/3] Generating {manifest_path}...")

    layer_types_int = [1 if t == "mamba" else (0 if t == "attention" else 2) for t in layers_block_type]

    manifest = {
        "model_config": {
            "model_name": "Nemotron-3.5-Lightning-30B-A3B-FrankensTin",
            "model_id": "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4",
            "architecture": "NEMOTRON_35",
            "vocab_size": config.get("vocab_size", 131072),
            "hidden_size": config.get("hidden_size", 2688),
            "num_hidden_layers": num_layers,
            "num_attention_heads": config.get("num_attention_heads", 32),
            "num_key_value_heads": config.get("num_key_value_heads", 2),
            "head_dim": config.get("head_dim", 128),
            "intermediate_size": config.get("intermediate_size", 1856),
            "moe_intermediate_size": config.get("moe_intermediate_size", 1856),
            "moe_shared_expert_intermediate_size": config.get("moe_shared_expert_intermediate_size", 3712),
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": config.get("num_experts_per_tok", 6),
            "routed_scaling_factor": config.get("routed_scaling_factor", 2.5),
            "rms_norm_eps": config.get("norm_eps", 1e-5),
            "rope_theta": config.get("rope_theta", 10000.0),
            "max_seq_len": 65536,
            "original_seq_len": 65536,
            "expert_dtype": "fp4",
            "ssm_d_state": config.get("ssm_state_size", 128),
            "ssm_d_conv": config.get("conv_kernel", 4),
            "ssm_expand": config.get("expand", 2),
            "ssm_dt_rank": config.get("mamba_num_heads", 64),
            "layer_types": layer_types_int,
            "layers_block_type": layers_block_type,
            "has_vision": True,
            "vision_delegate": "models/frankenstin/vision/moecher_manifest.json"
        },
        "dense_bin": "attention_dense_layers.bin",
        "expert_bin": "expert_weights.bin",
        "tokenizer": "tokenizer.json",
        "dense_tensors": dense_manifest,
        "expert_layout": {
            "block_size": EXPERT_BLOCK_SIZE,
            "n_layers": len(moe_layers),
            "n_experts": n_routed_experts,
            "parts": {
                p[0]: {
                    "offset_in_block": p[1],
                    "nbytes": p[2],
                    "dtype": p[3],
                    "shape": p[4]
                } for p in EXPERT_PARTS
            }
        }
    }

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"  [OK] Manifest created at {manifest_path}!")
    print("\n=================================================================")
    print("  NEMOTRON 3.5 LIGHTNING CONVERSION COMPLETE!")
    print(f"  Dense Bin:  {dense_bin_path} ({dense_gb:.2f} GB)")
    print(f"  Expert Bin: {expert_bin_path} ({expert_gb:.2f} GB)")
    print(f"  Manifest:   {manifest_path}")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
