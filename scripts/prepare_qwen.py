#!/usr/bin/env python3
"""
prepare_qwen.py — Download, extract, and build manifests for Qwen 3.8 (27B / 72B / 9B)
and Qwen 2.5 models for MinnieTheMoECher.

Usage:
    python3 scripts/prepare_qwen.py [--model-id Qwen/Qwen3.8-27B-Instruct]
                                   [--output-dir models/qwen3_8_27b]
                                   [--local-dir /path/to/safetensors]
"""

import os
import sys
import json
import struct
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

PAGE_SIZE = 4096

def read_safetensor_header(path: str) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json.decode("utf-8"))
    return header, 8 + header_size

def download_model(model_id: str, local_dir: Path) -> Path:
    if local_dir.exists() and any(local_dir.glob("*.safetensors")):
        print(f"[1/4] Found existing safetensors in {local_dir}")
        return local_dir

    if snapshot_download is None:
        raise RuntimeError("huggingface_hub is not installed. Please run: pip install huggingface_hub")

    print(f"[1/4] Downloading {model_id} via huggingface_hub to {local_dir}...")
    snapshot_download(
        repo_id=model_id,
        local_dir=str(local_dir),
        allow_patterns=["*.json", "*.safetensors", "tokenizer*"]
    )
    return local_dir

def prepare_qwen_model(model_dir: Path, out_dir: Path):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
        
    with open(config_file, "r") as f:
        raw_cfg = json.load(f)

    cfg = raw_cfg.get("text_config", raw_cfg)

    hidden_size = cfg.get("hidden_size", 5120)
    num_layers = cfg.get("num_hidden_layers", 64)
    num_heads = cfg.get("num_attention_heads", 24)
    kv_heads = cfg.get("num_key_value_heads", 4)
    head_dim = cfg.get("head_dim", 256)
    inter_size = cfg.get("intermediate_size", 17408)
    vocab_size = cfg.get("vocab_size", 248320)
    rope_theta = cfg.get("rope_theta", 1000000.0)
    max_pos = cfg.get("max_position_embeddings", 262144)
    
    print(f"[2/4] Model Architecture (Qwen 3.8):")
    print(f"      Layers: {num_layers}, Hidden: {hidden_size}, Heads: {num_heads} (KV: {kv_heads}), Head Dim: {head_dim}")
    print(f"      Intermediate: {inter_size}, Vocab: {vocab_size}, RoPE Theta: {rope_theta}, Max Context: {max_pos}")

    st_files = sorted(list(model_dir.glob("*.safetensors")))
    print(f"[3/4] Indexing {len(st_files)} safetensors files...")
    
    tensor_index = {}
    for st_path in st_files:
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

    dense_bin_path = out_dir / "attention_dense_layers.bin"
    manifest_path = out_dir / "moecher_manifest_qwen.json"
    
    print(f"[4/4] Extracting weights to {dense_bin_path}...")
    dense_tensors_meta = {}
    dense_offset = 0
    
    with open(dense_bin_path, "wb") as f_dense:
        for t_name, meta in tensor_index.items():
            with open(meta["file"], "rb") as f_in:
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

    print(f"      Wrote dense binary: {dense_bin_path} ({dense_offset / (1024**3):.2f} GB)")

    manifest = {
        "model_config": {
            "architecture": "qwen2",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": head_dim,
            "intermediate_size": inter_size,
            "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
            "rope_theta": rope_theta,
            "max_seq_len": min(max_pos, 32768),
            "original_seq_len": max_pos,
            "expert_dtype": "none",
            "n_routed_experts": 0,
            "num_experts_per_tok": 0,
            "n_shared_experts": 0,
            "n_hash_layers": 0
        },
        "tokenizer": {
            "tokenizer_json": "tokenizer.json"
        },
        "dense_bin": "attention_dense_layers.bin",
        "expert_bin": "",
        "dense_tensors": dense_tensors_meta,
        "expert_layout": {
            "block_size": 0,
            "n_layers": 0,
            "n_experts": 0,
            "parts": {}
        }
    }

    tok_src = model_dir / "tokenizer.json"
    if tok_src.exists():
        with open(out_dir / "tokenizer.json", "wb") as f_dst, open(tok_src, "rb") as f_src:
            f_dst.write(f_src.read())

    with open(manifest_path, "w") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"      Wrote manifest: {manifest_path}")
    print("\n[SUCCESS] Qwen model prepared successfully!")
    print(f"To run with MinnieTheMoECher:")
    print(f"  ./build/moecher --manifest {manifest_path} --max-vram 20 --quiet\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare Qwen 3.8 / 2.5 models for MinnieTheMoECher")
    parser.add_argument("--model-id", default="Qwen/Qwen2.5-Coder-32B-Instruct", help="HuggingFace model ID")
    parser.add_argument("--local-dir", default=None, help="Local directory if already downloaded")
    parser.add_argument("--output-dir", default="models/qwen", help="Output directory")
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    if args.local_dir:
        model_path = Path(args.local_dir).resolve()
    else:
        model_path = out_path / "raw_hf"
        download_model(args.model_id, model_path)

    prepare_qwen_model(model_path, out_path)
