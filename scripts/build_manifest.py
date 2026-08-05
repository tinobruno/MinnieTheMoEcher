#!/usr/bin/env python3
"""
build_manifest.py — Parse HuggingFace DeepSeek-V4-Flash safetensors and create:
  1. attention_dense_layers.bin  — All resident weights (attention, norms, embeddings, etc.)
  2. moe_experts.bin             — Expert weights, page-aligned for O_DIRECT
  3. moecher_manifest.json       — Tensor offsets, shapes, types, model config

Usage:
    python3 scripts/build_manifest.py [--model-dir <path>] [--output-dir <path>]
"""

import os
import sys
import json
import struct
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import re

PAGE_SIZE = 4096  # O_DIRECT alignment requirement

# ── safetensor helpers ──────────────────────────────────────────────────────────

def read_safetensor_header(path: str) -> Tuple[dict, int]:
    """Read JSON header from a safetensor file.
    Returns (header_dict, data_section_start_offset)."""
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json)
    return header, 8 + header_size


DTYPE_SIZES = {
    "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E8M0": 1,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "BOOL": 1, "U8": 1,
}


def tensor_nbytes(shape: list, dtype: str) -> int:
    n = 1
    for s in shape:
        n *= s
    return n * DTYPE_SIZES.get(dtype, 1)


def align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment


# ── classification ──────────────────────────────────────────────────────────────

def is_routed_expert(name: str) -> bool:
    """True for routed-expert tensors (NOT shared experts)."""
    return ".ffn.experts." in name and ".shared_experts." not in name


def parse_expert_key(name: str) -> Tuple[int, int, str]:
    """Extract (layer_id, expert_id, suffix) from e.g.
    'layers.5.ffn.experts.42.w1.weight' -> (5, 42, 'w1.weight')"""
    # Match both layers.N and mtp.N prefixes
    m = re.match(r"(?:layers|mtp)\.(\d+)\.ffn\.experts\.(\d+)\.(.+)", name)
    if not m:
        raise ValueError(f"Cannot parse expert key: {name}")
    return int(m.group(1)), int(m.group(2)), m.group(3)


# ── main logic ──────────────────────────────────────────────────────────────────

def build_manifest(model_dir: str, output_dir: str):
    model_dir = Path(model_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Find snapshot directory
    snapshot_dir = model_dir
    if not (model_dir / "config.json").exists():
        # Try HF cache layout
        snapshots = list(model_dir.glob("snapshots/*"))
        if snapshots:
            snapshot_dir = snapshots[0]
        else:
            print(f"ERROR: Cannot find config.json in {model_dir}", file=sys.stderr)
            sys.exit(1)

    print(f"Model directory: {snapshot_dir}")

    # Load model config
    with open(snapshot_dir / "config.json") as f:
        model_config = json.load(f)

    # Load inference config (has cleaner field names)
    inference_config = {}
    if (snapshot_dir / "inference" / "config.json").exists():
        with open(snapshot_dir / "inference" / "config.json") as f:
            inference_config = json.load(f)

    # Find all safetensor shards
    shard_files = sorted(snapshot_dir.glob("model-*-of-*.safetensors"))
    if not shard_files:
        print("ERROR: No safetensor shards found", file=sys.stderr)
        sys.exit(1)
    print(f"Found {len(shard_files)} safetensor shards")

    # ── Pass 1: Read all shard headers ───────────────────────────────────────

    # tensor_name -> {dtype, shape, shard_path, data_offset, nbytes}
    all_tensors: Dict[str, dict] = {}
    shard_data_starts: Dict[str, int] = {}  # shard_path -> data section start

    print("Reading shard headers...")
    for shard_path in shard_files:
        header, data_start = read_safetensor_header(str(shard_path))
        shard_data_starts[str(shard_path)] = data_start
        for name, info in header.items():
            if name == "__metadata__":
                continue
            start, end = info["data_offsets"]
            all_tensors[name] = {
                "dtype": info["dtype"],
                "shape": info["shape"],
                "shard_path": str(shard_path),
                "abs_offset": data_start + start,  # absolute byte offset in shard file
                "nbytes": end - start,
            }
    print(f"Total tensors: {len(all_tensors)}")

    # ── Classify tensors ─────────────────────────────────────────────────────

    resident_tensors = {}  # name -> tensor info
    expert_tensors = {}    # name -> tensor info

    for name, info in all_tensors.items():
        if is_routed_expert(name):
            expert_tensors[name] = info
        else:
            resident_tensors[name] = info

    print(f"Resident tensors: {len(resident_tensors)}")
    print(f"Expert tensors:   {len(expert_tensors)}")

    # ── Determine expert block layout ────────────────────────────────────────

    # All routed experts have the same shapes.  Verify by checking expert 0 of layer 0.
    n_layers = model_config.get("num_hidden_layers", inference_config.get("n_layers", 43))
    n_experts = model_config.get("n_routed_experts", 256)
    n_mtp = model_config.get("num_nextn_predict_layers", inference_config.get("n_mtp_layers", 1))

    # Expert sub-tensors in order: w1.weight, w1.scale, w3.weight, w3.scale, w2.weight, w2.scale
    EXPERT_PARTS = ["w1.weight", "w1.scale", "w3.weight", "w3.scale", "w2.weight", "w2.scale"]

    # Get shapes from layer 0, expert 0
    expert_part_info = {}
    for part in EXPERT_PARTS:
        key = f"layers.0.ffn.experts.0.{part}"
        if key in expert_tensors:
            t = expert_tensors[key]
            expert_part_info[part] = {"dtype": t["dtype"], "shape": t["shape"], "nbytes": t["nbytes"]}
        else:
            print(f"WARNING: Missing {key}", file=sys.stderr)

    # Compute per-expert block size
    expert_raw_size = sum(p["nbytes"] for p in expert_part_info.values())
    expert_block_size = align_up(expert_raw_size, PAGE_SIZE)
    print(f"Expert raw size: {expert_raw_size} bytes")
    print(f"Expert block size (page-aligned): {expert_block_size} bytes")

    # Compute sub-offsets within each expert block
    expert_part_offsets = {}
    off = 0
    for part in EXPERT_PARTS:
        expert_part_offsets[part] = off
        off += expert_part_info[part]["nbytes"]

    # Total number of expert blocks: main layers + mtp layers
    # MTP layers also have experts
    total_expert_layers = n_layers  # We'll handle MTP experts separately
    mtp_expert_layers = []
    for name in expert_tensors:
        if name.startswith("mtp."):
            m = re.match(r"mtp\.(\d+)\.", name)
            if m:
                mtp_layer = int(m.group(1))
                if mtp_layer not in mtp_expert_layers:
                    mtp_expert_layers.append(mtp_layer)
    mtp_expert_layers.sort()
    print(f"MTP expert layers: {mtp_expert_layers}")

    # Total blocks = (n_layers + len(mtp_expert_layers)) * n_experts
    total_layers_with_experts = n_layers + len(mtp_expert_layers)
    total_expert_blocks = total_layers_with_experts * n_experts

    print(f"Total expert blocks: {total_expert_blocks}")
    total_expert_bytes = total_expert_blocks * expert_block_size
    print(f"Total moe_experts.bin size: {total_expert_bytes / 1e9:.2f} GB")

    # ── Pass 2: Write attention_dense_layers.bin ─────────────────────────────

    dense_path = output_dir / "attention_dense_layers.bin"
    print(f"\nWriting {dense_path}...")

    dense_manifest = {}  # name -> {offset, nbytes, dtype, shape}
    dense_offset = 0

    # Sort resident tensors for deterministic output
    sorted_resident = sorted(resident_tensors.keys())

    t0 = time.time()
    with open(dense_path, "wb") as out_f:
        for i, name in enumerate(sorted_resident):
            info = resident_tensors[name]
            # Read tensor data from shard
            with open(info["shard_path"], "rb") as shard_f:
                shard_f.seek(info["abs_offset"])
                data = shard_f.read(info["nbytes"])
            assert len(data) == info["nbytes"], f"Short read for {name}"

            # Write to dense bin
            out_f.write(data)
            dense_manifest[name] = {
                "offset": dense_offset,
                "nbytes": info["nbytes"],
                "dtype": info["dtype"],
                "shape": info["shape"],
            }
            dense_offset += info["nbytes"]

            if (i + 1) % 200 == 0 or i == len(sorted_resident) - 1:
                elapsed = time.time() - t0
                print(f"  [{i+1}/{len(sorted_resident)}] {dense_offset/1e9:.2f} GB written ({elapsed:.1f}s)")

    print(f"  Dense bin total: {dense_offset / 1e9:.2f} GB")

    # ── Pass 3: Write moe_experts.bin ────────────────────────────────────────

    experts_path = output_dir / "moe_experts.bin"
    print(f"\nWriting {experts_path}...")

    # Build layer index: maps logical layer index (0..total_layers_with_experts-1)
    # to the prefix used in tensor names
    layer_prefixes = []
    for l in range(n_layers):
        layer_prefixes.append(f"layers.{l}")
    for ml in mtp_expert_layers:
        layer_prefixes.append(f"mtp.{ml}")

    expert_manifest = {
        "block_size": expert_block_size,
        "n_layers": total_layers_with_experts,
        "n_experts": n_experts,
        "parts": {},
        "part_order": EXPERT_PARTS,
        "layer_prefixes": layer_prefixes,
    }
    for part in EXPERT_PARTS:
        expert_manifest["parts"][part] = {
            "offset_in_block": expert_part_offsets[part],
            "nbytes": expert_part_info[part]["nbytes"],
            "dtype": expert_part_info[part]["dtype"],
            "shape": expert_part_info[part]["shape"],
        }

    padding_bytes = expert_block_size - expert_raw_size
    zero_pad = b"\x00" * padding_bytes

    t0 = time.time()
    with open(experts_path, "wb") as out_f:
        block_idx = 0
        for layer_idx, prefix in enumerate(layer_prefixes):
            for expert_id in range(n_experts):
                # Write each part
                for part in EXPERT_PARTS:
                    tensor_name = f"{prefix}.ffn.experts.{expert_id}.{part}"
                    if tensor_name not in expert_tensors:
                        # Missing tensor — write zeros
                        nbytes = expert_part_info[part]["nbytes"]
                        out_f.write(b"\x00" * nbytes)
                        continue
                    info = expert_tensors[tensor_name]
                    with open(info["shard_path"], "rb") as shard_f:
                        shard_f.seek(info["abs_offset"])
                        data = shard_f.read(info["nbytes"])
                    assert len(data) == info["nbytes"], f"Short read for {tensor_name}"
                    out_f.write(data)

                # Page-align padding
                if padding_bytes > 0:
                    out_f.write(zero_pad)

                block_idx += 1
                if block_idx % 1000 == 0:
                    elapsed = time.time() - t0
                    pct = block_idx / total_expert_blocks * 100
                    written_gb = block_idx * expert_block_size / 1e9
                    print(f"  [{block_idx}/{total_expert_blocks}] {pct:.1f}% — {written_gb:.2f} GB ({elapsed:.1f}s)")

    elapsed = time.time() - t0
    print(f"  Expert bin total: {total_expert_bytes / 1e9:.2f} GB ({elapsed:.1f}s)")

    # ── Write manifest JSON ──────────────────────────────────────────────────

    # Extract key model hyperparameters for the C++ engine
    engine_config = {
        "vocab_size": model_config.get("vocab_size", 129280),
        "hidden_size": model_config.get("hidden_size", 4096),
        "num_hidden_layers": n_layers,
        "num_attention_heads": model_config.get("num_attention_heads", 64),
        "num_key_value_heads": model_config.get("num_key_value_heads", 1),
        "head_dim": model_config.get("head_dim", 512),
        "qk_rope_head_dim": model_config.get("qk_rope_head_dim", 64),
        "q_lora_rank": model_config.get("q_lora_rank", 1024),
        "o_lora_rank": model_config.get("o_lora_rank", 1024),
        "o_groups": model_config.get("o_groups", 8),
        "moe_intermediate_size": model_config.get("moe_intermediate_size", 2048),
        "n_routed_experts": n_experts,
        "num_experts_per_tok": model_config.get("num_experts_per_tok", 6),
        "n_shared_experts": model_config.get("n_shared_experts", 1),
        "n_hash_layers": model_config.get("num_hash_layers", inference_config.get("n_hash_layers", 3)),
        "rms_norm_eps": model_config.get("rms_norm_eps", 1e-6),
        "rope_theta": model_config.get("rope_theta", 10000.0),
        "rope_factor": model_config.get("rope_scaling", {}).get("factor", inference_config.get("rope_factor", 16)),
        "rope_beta_fast": model_config.get("rope_scaling", {}).get("beta_fast", 32),
        "rope_beta_slow": model_config.get("rope_scaling", {}).get("beta_slow", 1),
        "original_seq_len": model_config.get("rope_scaling", {}).get("original_max_position_embeddings",
                               inference_config.get("original_seq_len", 65536)),
        "sliding_window": model_config.get("sliding_window", 128),
        "scoring_func": model_config.get("scoring_func", "sqrtsoftplus"),
        "routed_scaling_factor": model_config.get("routed_scaling_factor", 1.5),
        "swiglu_limit": model_config.get("swiglu_limit", 10.0),
        "hc_mult": model_config.get("hc_mult", 4),
        "hc_sinkhorn_iters": model_config.get("hc_sinkhorn_iters", 20),
        "hc_eps": model_config.get("hc_eps", 1e-6),
        "bos_token_id": model_config.get("bos_token_id", 0),
        "eos_token_id": model_config.get("eos_token_id", 1),
        "compress_ratios": model_config.get("compress_ratios", inference_config.get("compress_ratios", [])),
        "compress_rope_theta": model_config.get("compress_rope_theta", inference_config.get("compress_rope_theta", 160000)),
        "index_n_heads": model_config.get("index_n_heads", 64),
        "index_head_dim": model_config.get("index_head_dim", 128),
        "index_topk": model_config.get("index_topk", 512),
        "expert_dtype": model_config.get("expert_dtype", "fp4"),
        "torch_dtype": model_config.get("torch_dtype", "bfloat16"),
    }

    # Tokenizer special token IDs
    tokenizer_info = {
        "bos_token_id": 0,
        "eos_token_id": 1,
        "user_token": "<\uff5cUser\uff5c>",
        "assistant_token": "<\uff5cAssistant\uff5c>",
        "tokenizer_json": str(snapshot_dir / "tokenizer.json"),
    }

    manifest = {
        "model_config": engine_config,
        "tokenizer": tokenizer_info,
        "dense_bin": str(dense_path.resolve()),
        "expert_bin": str(experts_path.resolve()),
        "dense_tensors": dense_manifest,
        "expert_layout": expert_manifest,
    }

    manifest_path = output_dir / "moecher_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)
    print(f"\nManifest written to {manifest_path}")
    print("Done!")


# ── CLI ─────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Build moecher manifest from HF safetensors")
    parser.add_argument("--model-dir", type=str,
                        default=os.path.expanduser(
                            "~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash"),
                        help="Path to HF model cache directory")
    parser.add_argument("--output-dir", type=str, default=".",
                        help="Output directory for bin files and manifest")
    args = parser.parse_args()
    build_manifest(args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
