#!/usr/bin/env python3
"""
prepare_nemotron_bolt.py — Fast Zero-Copy Converter for NVIDIA Nemotron 3.5 Bolt (74B-A9B) NVFP4
Architecture: nemotron_h_puzzle / LatentMoE + Mamba-2 SSM

Produces:
  1. attention_dense_layers.bin  — All resident weights:
                                   • Embeddings + Norms + LM Head
                                   • 40 Mamba-2 SSM mixer layers (128 heads, 18568 in-proj)
                                   • 8 GQA Attention layers (32 heads, 2 KV heads)
                                   • 40 LatentMoE bottlenecks (fc1_latent_proj 4096->1024, fc2_latent_proj 1024->4096)
                                   • 40 Shared Experts (5376 intermediate)
                                   • 40 Router Gates (128 experts, sigmoid + bias)
  2. expert_weights.bin          — Page-aligned (4096-byte) routed LatentMoE experts (1024->moe_inter->1024)
  3. expert_scales.bin           — Per-expert float32 scales (up and down)
  4. moecher_manifest_24gb.json  — Manifest for 24GB VRAM target (100% in-VRAM resident execution)
  5. moecher_manifest_16gb.json  — Manifest for 16GB VRAM target (18 MoE layers in VRAM, 22 in Host DRAM streaming)
"""

import os
import sys
import json
import time
import struct
import re
from pathlib import Path

PAGE_SIZE = 4096  # Direct I/O / O_DIRECT requirement

def read_safetensor_header(path):
    with open(path, "rb") as f:
        hlen = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(hlen))
    return header, 8 + hlen

def compute_expert_block_layout(latent_size=1024, moe_inter=2048):
    up_w_bytes = moe_inter * (latent_size // 2)      # 2048 * 512 = 1,048,576
    up_s_bytes = moe_inter * (latent_size // 16)     # 2048 * 64  =   131,072
    down_w_bytes = latent_size * (moe_inter // 2)    # 1024 * 1024 = 1,048,576
    down_s_bytes = latent_size * (moe_inter // 16)   # 1024 * 128  =   131,072

    up_w_off = 0
    up_s_off = up_w_off + up_w_bytes
    up_s2_off = up_s_off + up_s_bytes
    down_w_off = up_s2_off + 16  # 16-byte alignment
    down_s_off = down_w_off + down_w_bytes
    down_s2_off = down_s_off + down_s_bytes
    raw_size = down_s2_off + 16

    block_size = ((raw_size + PAGE_SIZE - 1) // PAGE_SIZE) * PAGE_SIZE

    parts = [
        ("up_proj.weight", up_w_off, up_w_bytes, "fp4", [moe_inter, latent_size // 2]),
        ("up_proj.weight_scale", up_s_off, up_s_bytes, "F8_E4M3", [moe_inter, latent_size // 16]),
        ("up_proj.weight_scale_2", up_s2_off, 4, "F32", []),
        ("down_proj.weight", down_w_off, down_w_bytes, "fp4", [latent_size, moe_inter // 2]),
        ("down_proj.weight_scale", down_s_off, down_s_bytes, "F8_E4M3", [latent_size, moe_inter // 16]),
        ("down_proj.weight_scale_2", down_s2_off, 4, "F32", []),
    ]
    return block_size, parts

def main():
    raw_dir = Path(sys.argv[1] if len(sys.argv) > 1 else "E:/dev/nemotron_bolt_raw")
    out_dir = Path(sys.argv[2] if len(sys.argv) > 2 else "models/nemotron_3.5_bolt")
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=================================================================")
    print("  Nemotron 3.5 Bolt (74B-A9B) LatentMoE Converter")
    print(f"  Source: {raw_dir}")
    print(f"  Target: {out_dir}")
    print("=================================================================\n")

    config_path = raw_dir / "config.json"
    index_path = raw_dir / "model.safetensors.index.json"

    if not config_path.exists():
        print(f"Generating default Bolt-74B configuration template...")
        config = {
            "model_name": "Nemotron-3.5-Bolt-74B-A9B",
            "model_id": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4",
            "architecture": "nemotron_h_puzzle",
            "vocab_size": 131072,
            "hidden_size": 4096,
            "num_hidden_layers": 88,
            "num_attention_heads": 32,
            "num_key_value_heads": 2,
            "head_dim": 128,
            "moe_latent_size": 1024,
            "moe_intermediate_size": 2048,
            "moe_shared_expert_intermediate_size": 5376,
            "n_routed_experts": 128,
            "num_experts_per_tok": 9,
            "routed_scaling_factor": 5.0,
            "norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "max_seq_len": 65536,
            "original_seq_len": 65536,
            "expert_dtype": "fp4",
            "ssm_d_state": 128,
            "ssm_d_conv": 4,
            "ssm_expand": 2,
            "mamba_num_heads": 128,
            "mamba_head_dim": 64,
        }
    else:
        with open(config_path, "r", encoding="utf-8") as f:
            config = json.load(f)

    layers_block_type = config.get("layers_block_type", [])
    if not layers_block_type:
        # Default Nemotron-H Puzzle 88-layer pattern: alternating Mamba and LatentMoE with periodic Attention
        # 40 Mamba, 40 LatentMoE, 8 Attention = 88 total
        layers_block_type = []
        attn_positions = {3, 11, 23, 35, 47, 59, 71, 83}
        non_attn_toggle = 0
        for l in range(88):
            if l in attn_positions:
                layers_block_type.append("attention")
            elif non_attn_toggle % 2 == 0:
                layers_block_type.append("mamba")
                non_attn_toggle += 1
            else:
                layers_block_type.append("moe")
                non_attn_toggle += 1

    num_layers = len(layers_block_type)
    moe_layers = [l for l, t in enumerate(layers_block_type) if t == "moe"]
    n_routed_experts = config.get("n_routed_experts", 128)
    latent_size = config.get("moe_latent_size", 1024)
    moe_inter = config.get("moe_intermediate_size", 2048)

    expert_block_size, expert_parts = compute_expert_block_layout(latent_size, moe_inter)

    print(f"Total layers: {num_layers} ({len(moe_layers)} MoE layers, {sum(1 for t in layers_block_type if t == 'mamba')} Mamba, {sum(1 for t in layers_block_type if t == 'attention')} Attention)")
    print(f"LatentMoE Bottleneck: {config.get('hidden_size', 4096)} -> {latent_size} -> {moe_inter}")
    print(f"Total routed experts: {len(moe_layers)} * {n_routed_experts} = {len(moe_layers) * n_routed_experts}")
    print(f"Expert block size: {expert_block_size} bytes ({expert_block_size / 1024 / 1024:.2f} MB aligned to {PAGE_SIZE})")

    has_raw_shards = index_path.exists()
    dense_manifest = {}
    total_experts = len(moe_layers) * n_routed_experts

    if has_raw_shards:
        with open(index_path, "r", encoding="utf-8") as f:
            index_data = json.load(f)
        weight_map = index_data.get("weight_map", {})
        print(f"Found {len(weight_map)} tensors in model index.")

        moe_layer_to_idx = {l: i for i, l in enumerate(moe_layers)}
        dense_shard_tensors = {}
        expert_shard_tensors = {}
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

        # 1. Write attention_dense_layers.bin
        dense_bin_path = out_dir / "attention_dense_layers.bin"
        print(f"\n[1/3] Writing {dense_bin_path}...")
        t0 = time.time()
        current_dense_offset = 0

        with open(dense_bin_path, "wb") as dense_f:
            for shard in sorted(dense_shard_tensors.keys()):
                shard_path = raw_dir / shard
                header, data_start = read_safetensor_header(shard_path)
                with open(shard_path, "rb") as sf:
                    for tensor_name in sorted(dense_shard_tensors[shard]):
                        tmeta = header[tensor_name]
                        offsets = tmeta["data_offsets"]
                        nbytes = offsets[1] - offsets[0]
                        sf.seek(data_start + offsets[0])
                        dense_f.write(sf.read(nbytes))
                        dense_manifest[tensor_name] = {
                            "offset": current_dense_offset,
                            "nbytes": nbytes,
                            "dtype": tmeta.get("dtype", "BF16"),
                            "shape": tmeta.get("shape", [])
                        }
                        current_dense_offset += nbytes
        print(f"  [OK] Done! Dense bin size: {current_dense_offset / (1024**3):.2f} GB in {time.time() - t0:.1f}s")

        # 2. Write expert_weights.bin
        expert_bin_path = out_dir / "expert_weights.bin"
        total_expert_bytes = total_experts * expert_block_size
        print(f"\n[2/3] Writing {expert_bin_path} ({total_expert_bytes / (1024**3):.2f} GB)...")
        t0 = time.time()
        part_offsets = {p[0]: (p[1], p[2]) for p in expert_parts}

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
                        block_idx = moe_idx * n_routed_experts + expert_id
                        file_pos = block_idx * expert_block_size + part_off
                        exp_f.seek(file_pos)
                        exp_f.write(sf.read(nbytes))
                        written_count += 1
        print(f"  [OK] Done! Expert bin written in {time.time() - t0:.1f}s")
    else:
        print("Note: Raw safetensors not provided. Generating manifest blueprints for 24GB and 16GB targets...")

    # Write default expert_scales.bin (1.0f)
    scales_path = out_dir / "expert_scales.bin"
    if not scales_path.exists():
        with open(scales_path, "wb") as sf:
            sf.write(struct.pack(f"<{total_experts * 2}f", *([1.0] * (total_experts * 2))))
        print(f"  [OK] Initialized unit expert scales at {scales_path}")

    # Generate Manifests:
    layer_types_int = [1 if t == "mamba" else (0 if t == "attention" else 2) for t in layers_block_type]

    base_manifest = {
        "model_config": {
            "model_name": "Nemotron-3.5-Bolt-74B-A9B",
            "model_id": "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4",
            "architecture": "nemotron_h_puzzle",
            "vocab_size": config.get("vocab_size", 131072),
            "hidden_size": config.get("hidden_size", 4096),
            "num_hidden_layers": num_layers,
            "num_attention_heads": config.get("num_attention_heads", 32),
            "num_key_value_heads": config.get("num_key_value_heads", 2),
            "head_dim": config.get("head_dim", 128),
            "intermediate_size": config.get("intermediate_size", 2048),
            "moe_intermediate_size": moe_inter,
            "moe_latent_size": latent_size,
            "moe_shared_expert_intermediate_size": config.get("moe_shared_expert_intermediate_size", 5376),
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": config.get("num_experts_per_tok", 9),
            "routed_scaling_factor": config.get("routed_scaling_factor", 5.0),
            "rms_norm_eps": config.get("norm_eps", 1e-5),
            "rope_theta": config.get("rope_theta", 10000.0),
            "max_seq_len": 65536,
            "original_seq_len": 65536,
            "expert_dtype": "fp4",
            "ssm_d_state": config.get("ssm_state_size", 128),
            "ssm_d_conv": config.get("conv_kernel", 4),
            "ssm_expand": config.get("expand", 2),
            "mamba_num_heads": config.get("mamba_num_heads", 128),
            "mamba_head_dim": config.get("mamba_head_dim", 64),
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
            "block_size": expert_block_size,
            "n_layers": len(moe_layers),
            "n_experts": n_routed_experts,
            "parts": {
                p[0]: {
                    "offset_in_block": p[1],
                    "nbytes": p[2],
                    "dtype": p[3],
                    "shape": p[4]
                } for p in expert_parts
            }
        }
    }

    # 1. 24GB Target Manifest (100% VRAM Resident)
    manifest_24gb = json.loads(json.dumps(base_manifest))
    manifest_24gb["target_hardware"] = {
        "vram_gb": 24,
        "resident_moe_layers": 40,
        "dram_stream_moe_layers": 0,
        "description": "100% VRAM resident execution for RTX 3090 / 4090 / 5090 (24GB VRAM)"
    }
    path_24gb = out_dir / "moecher_manifest_24gb.json"
    with open(path_24gb, "w", encoding="utf-8") as f:
        json.dump(manifest_24gb, f, indent=2)
    print(f"  [OK] Created 24GB Manifest: {path_24gb}")

    # 2. 16GB Target Manifest (Hybrid 18 VRAM / 22 DRAM Lookahead Stream)
    manifest_16gb = json.loads(json.dumps(base_manifest))
    manifest_16gb["target_hardware"] = {
        "vram_gb": 16,
        "resident_moe_layers": 18,
        "dram_stream_moe_layers": 22,
        "description": "Hybrid VRAM + DRAM Streaming for RTX 4080 / 5060 Ti (16GB VRAM)"
    }
    path_16gb = out_dir / "moecher_manifest_16gb.json"
    with open(path_16gb, "w", encoding="utf-8") as f:
        json.dump(manifest_16gb, f, indent=2)
    print(f"  [OK] Created 16GB Manifest: {path_16gb}")

    print("\n=================================================================")
    print("  NEMOTRON 3.5 BOLT CONVERTER & MANIFESTS READY!")
    print("=================================================================\n")

if __name__ == "__main__":
    main()
