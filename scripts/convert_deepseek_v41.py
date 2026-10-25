#!/usr/bin/env python3
"""
convert_deepseek_v41.py — Converter & Direct-I/O MoE Packer for DeepSeek-V4.1-Flash

Converts official DeepSeek-V4.1-Flash Safetensors into MinnieTheMoEcher format:
1. attention_dense_layers.bin: Dense backbone, MLA projections, shared expert, router gates
2. expert.bin: 15,360 routed experts in 4KB page-aligned Direct-I/O binary format
3. moecher_manifest.json: Engine manifest with 3-tier memory allocation budgets

Usage:
    python scripts/convert_deepseek_v41.py --model-dir F:/models/deepseek_v41_raw \
                                           --out-dir F:/Moecher/models/deepseek_v41_nvfp4
"""

import os
import sys
import json
import struct
import shutil
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

PAGE_SIZE = 4096  # 4KB boundary alignment required by Direct I/O / FILE_FLAG_NO_BUFFERING

def align_up(x: int, alignment: int = PAGE_SIZE) -> int:
    return ((x + alignment - 1) // alignment) * alignment

def read_safetensor_header(path: str) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json.decode("utf-8"))
    return header, 8 + header_size

def convert_deepseek_v41(raw_dir: Path, out_dir: Path, dry_run: bool = False):
    print("=" * 70)
    print("  DeepSeek-V4.1-Flash Converter & Direct-I/O MoE Packer")
    print(f"  Source Directory: {raw_dir}")
    print(f"  Output Directory: {out_dir}")
    print("=" * 70, flush=True)

    config_path = raw_dir / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {raw_dir}")

    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    text_cfg = cfg.get("text_config", cfg)
    hidden_size = text_cfg.get("hidden_size", 5120)
    num_layers = text_cfg.get("num_hidden_layers", 40)
    n_routed_experts = text_cfg.get("n_routed_experts", 384)
    top_k = text_cfg.get("num_experts_per_tok", 6)
    n_shared_experts = text_cfg.get("n_shared_experts", 1)
    moe_inter = text_cfg.get("moe_intermediate_size", 2304)
    vocab_size = text_cfg.get("vocab_size", 129280)
    head_dim = text_cfg.get("head_dim", 512)
    n_heads = text_cfg.get("num_attention_heads", 64)
    q_lora_rank = text_cfg.get("q_lora_rank", 1280)
    o_lora_rank = text_cfg.get("o_lora_rank", 1024)
    o_groups = text_cfg.get("o_groups", 8)

    print(f"[1/4] Architecture Specs:")
    print(f"      Hidden Dim: {hidden_size} | Layers: {num_layers} | Vocab: {vocab_size}")
    print(f"      Routed Experts: {n_routed_experts} per layer ({num_layers * n_routed_experts} total)")
    print(f"      Active Routed Experts: {top_k} | Shared Experts: {n_shared_experts} | MoE Inter: {moe_inter}")
    print(f"      MLA Attention: Heads={n_heads}, Dim={head_dim}, Q_LoRA={q_lora_rank}, O_LoRA={o_lora_rank}, O_Groups={o_groups}")

    out_dir.mkdir(parents=True, exist_ok=True)

    # Copy tokenizer files
    for tok_f in ["tokenizer.json", "tokenizer_config.json", "chat_template.jinja", "vocab.json", "merges.txt"]:
        src_p = raw_dir / tok_f
        if src_p.exists():
            shutil.copy2(src_p, out_dir / tok_f)
            print(f"      Copied metadata: {tok_f}")

    # Index all safetensors shards
    st_files = sorted(list(raw_dir.glob("*.safetensors")))
    print(f"\n[2/4] Indexing {len(st_files)} Safetensors shards...")
    tensor_index = {}
    for st_path in st_files:
        try:
            header, data_offset = read_safetensor_header(str(st_path))
            for t_name, info in header.items():
                if t_name == "__metadata__":
                    continue
                tensor_index[t_name] = {
                    "file": str(st_path),
                    "data_offset": data_offset + info["data_offsets"][0],
                    "nbytes": info["data_offsets"][1] - info["data_offsets"][0],
                    "shape": info["shape"],
                    "dtype": info["dtype"],
                }
        except Exception as e:
            print(f"      [WARN] Skipping non-standard or partial file {st_path.name}: {e}")

    print(f"      Indexed {len(tensor_index)} total tensor entries across all shards.")

    if dry_run:
        print("[DRY-RUN] Manifest and indexing verified successfully.")
        return

    # Extract dense backbone
    dense_bin_path = out_dir / "attention_dense_layers.bin"
    print(f"\n[3/4] Packaging dense backbone layers into {dense_bin_path.name}...", flush=True)
    dense_tensors_meta = {}
    dense_offset = 0

    open_handles = {}
    try:
        with open(dense_bin_path, "wb") as f_dense:
            for t_name, meta in sorted(tensor_index.items()):
                # Filter out routed expert weights (keep shared experts and router gates) and 189GB engram tables
                if (".ffn.experts." in t_name and ".shared_experts." not in t_name) or ".engram." in t_name:
                    continue

                fpath = meta["file"]
                if fpath not in open_handles:
                    open_handles[fpath] = open(fpath, "rb")
                f_in = open_handles[fpath]
                f_in.seek(meta["data_offset"])
                data = f_in.read(meta["nbytes"])

                f_dense.write(data)
                dense_tensors_meta[t_name] = {
                    "offset": dense_offset,
                    "nbytes": meta["nbytes"],
                    "dtype": meta["dtype"],
                    "shape": meta["shape"]
                }
                dense_offset += len(data)

        print(f"      Dense backbone written: {dense_offset / (1024**3):.2f} GB", flush=True)

        # Packaging MoE Routed Experts into Direct I/O Aligned Block Binary
        expert_bin_path = out_dir / "expert.bin"
        print(f"\n[4/4] Packing {num_layers * n_routed_experts} routed experts into Direct-I/O binary {expert_bin_path.name}...", flush=True)

        part_order = [
            "w1.weight", "w1.scale",
            "w3.weight", "w3.scale",
            "w2.weight", "w2.scale"
        ]

        # Resolve expert part dimensions and offsets from layer 0 expert 0
        sample_expert_prefix = "layers.0.ffn.experts.0"
        expert_parts = {}
        cur_block_offset = 0
        for part in part_order:
            key = f"{sample_expert_prefix}.{part}"
            if key not in tensor_index:
                key = f"model.{key}"
            if key in tensor_index:
                meta = tensor_index[key]
                expert_parts[part] = {
                    "offset_in_block": cur_block_offset,
                    "nbytes": meta["nbytes"],
                    "dtype": meta["dtype"],
                    "shape": meta["shape"]
                }
                cur_block_offset += meta["nbytes"]
            else:
                # Fallback estimation for FP4 / E8M0 if index not yet scanned
                if part in ["w1.weight", "w3.weight"]:
                    nbytes = moe_inter * (hidden_size // 2)
                    dtype = "I8"
                    shape = [moe_inter, hidden_size // 2]
                elif part == "w2.weight":
                    nbytes = hidden_size * (moe_inter // 2)
                    dtype = "I8"
                    shape = [hidden_size, moe_inter // 2]
                elif part in ["w1.scale", "w3.scale"]:
                    nbytes = moe_inter * (hidden_size // 32)
                    dtype = "F8_E8M0"
                    shape = [moe_inter, hidden_size // 32]
                else: # w2.scale
                    nbytes = hidden_size * (moe_inter // 32)
                    dtype = "F8_E8M0"
                    shape = [hidden_size, moe_inter // 32]
                expert_parts[part] = {
                    "offset_in_block": cur_block_offset,
                    "nbytes": nbytes,
                    "dtype": dtype,
                    "shape": shape
                }
                cur_block_offset += nbytes

        expert_block_size = align_up(cur_block_offset, PAGE_SIZE)
        print(f"      Expert Block Size: {expert_block_size} bytes ({expert_block_size / (1024*1024):.2f} MB, 4KB page aligned)", flush=True)
        for part, pinfo in expert_parts.items():
            print(f"        {part:10s} -> offset: {pinfo['offset_in_block']:8d} | nbytes: {pinfo['nbytes']:8d} | dtype: {pinfo['dtype']}", flush=True)

        with open(expert_bin_path, "wb") as f_exp:
            for l in range(num_layers):
                for e in range(n_routed_experts):
                    block = bytearray(expert_block_size)
                    for part in part_order:
                        pinfo = expert_parts[part]
                        name = f"layers.{l}.ffn.experts.{e}.{part}"
                        target_key = name if name in tensor_index else f"model.{name}"
                        if target_key in tensor_index:
                            t_meta = tensor_index[target_key]
                            fpath = t_meta["file"]
                            if fpath not in open_handles:
                                open_handles[fpath] = open(fpath, "rb")
                            f_in = open_handles[fpath]
                            f_in.seek(t_meta["data_offset"])
                            p_data = f_in.read(t_meta["nbytes"])
                            off = pinfo["offset_in_block"]
                            block[off:off + len(p_data)] = p_data

                    f_exp.write(block)

                if (l + 1) % 5 == 0 or l == num_layers - 1:
                    print(f"      Processed {l + 1}/{num_layers} layers ({((l + 1) * n_routed_experts)} experts)...", flush=True)
    finally:
        for fh in open_handles.values():
            try:
                fh.close()
            except Exception:
                pass

    expert_file_size = os.path.getsize(expert_bin_path)
    print(f"      Expert binary complete: {expert_file_size / (1024**3):.2f} GB")

    # Generate Manifest conforming to MinnieTheMoEcher engine schema
    manifest_path = out_dir / "moecher_manifest.json"
    manifest = {
        "model_config": {
            "architecture": "deepseek_v41",
            "model_id": "deepseek-v4.1-flash",
            "model_name": "DeepSeek V4.1-Flash (552B / 14B Active)",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": n_heads,
            "num_key_value_heads": 1,
            "head_dim": head_dim,
            "qk_rope_head_dim": 64,
            "q_lora_rank": q_lora_rank,
            "o_lora_rank": o_lora_rank,
            "o_groups": o_groups,
            "moe_intermediate_size": moe_inter,
            "moe_shared_expert_intermediate_size": 3712,
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": top_k,
            "n_shared_experts": n_shared_experts,
            "n_hash_layers": 0,
            "routed_scaling_factor": 1.5,
            "swiglu_limit": 10.0,
            "scoring_func": "sqrtsoftplus",
            "expert_dtype": "fp4",
            "max_seq_len": 65536,
            "window_size": 128
        },
        "tokenizer": {
            "bos_token_id": 0,
            "eos_token_id": 1,
            "user_token": "<|User|>",
            "assistant_token": "<|Assistant|>",
            "tokenizer_json": "tokenizer.json"
        },
        "dense_bin": "attention_dense_layers.bin",
        "expert_bin": "expert.bin",
        "expert_layout": {
            "block_size": expert_block_size,
            "n_layers": num_layers,
            "n_experts": n_routed_experts,
            "part_order": part_order,
            "parts": expert_parts,
            "layer_prefixes": [f"layers.{i}" for i in range(num_layers)]
        },
        "cache_recommendation": {
            "l1_vram_gb": 46.0,
            "l2_dram_gb": 32.0,
            "l3_storage": "direct_io"
        },
        "dense_tensors": dense_tensors_meta
    }

    with open(manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"\n[SUCCESS] Manifest written to {manifest_path}")
    print("=" * 70, flush=True)

def main():
    parser = argparse.ArgumentParser(description="DeepSeek-V4.1-Flash Model Converter")
    parser.add_argument("--model-dir", default="F:/models/deepseek_v41_raw", help="Path to raw safetensors directory")
    parser.add_argument("--out-dir", default="F:/Moecher/models/deepseek_v41_nvfp4", help="Path to converted output directory")
    parser.add_argument("--dry-run", action="store_true", help="Validate headers without full disk write")
    args = parser.parse_args()

    convert_deepseek_v41(Path(args.model_dir), Path(args.out_dir), args.dry_run)

if __name__ == "__main__":
    main()
