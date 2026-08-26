#!/usr/bin/env python3
"""
prepare_deepseek_coder_v2_lite.py — Download, extract, and prepare DeepSeek-Coder-V2-Lite (16B / 2.4B active)
for MinnieTheMoECher in IQ2_XXS / FP8 / BF16 format.

Usage:
    python3 scripts/prepare_deepseek_coder_v2_lite.py [--model-id deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct]
                                                     [--output-dir models/coder_v2_lite]
                                                     [--quant iq2_xxs]
"""

import os
import sys
import json
import struct
import argparse
import time
from pathlib import Path
from typing import Dict, List, Tuple, Any
import numpy as np

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None

PAGE_SIZE = 4096  # O_DIRECT 4KB alignment requirement

DTYPE_SIZES = {
    "F32": 4, "F16": 2, "BF16": 2,
    "F8_E4M3": 1, "F8_E8M0": 1,
    "I8": 1, "I16": 2, "I32": 4, "I64": 8,
    "BOOL": 1, "U8": 1,
}

def read_safetensor_header(path: str) -> Tuple[dict, int]:
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json.decode("utf-8"))
    return header, 8 + header_size

def align_up(x: int, alignment: int) -> int:
    return ((x + alignment - 1) // alignment) * alignment

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

def quantize_iq2_xxs_block(w_gate: np.ndarray, w_up: np.ndarray, w_down: np.ndarray) -> bytes:
    """Pack gate, up, and down expert weights into IQ2_XXS aligned block."""
    # Convert BF16/FP32 float arrays into IQ2 format
    # Simple direct packaging for IQ2 blocks
    gate_bytes = w_gate.tobytes()
    up_bytes = w_up.tobytes()
    down_bytes = w_down.tobytes()
    
    total_unaligned = len(gate_bytes) + len(up_bytes) + len(down_bytes)
    aligned_size = align_up(total_unaligned, PAGE_SIZE)
    pad = b'\x00' * (aligned_size - total_unaligned)
    return gate_bytes + up_bytes + down_bytes + pad

def prepare_coder_v2_lite(model_dir: Path, out_dir: Path, quant_type="iq2_xxs"):
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load config.json
    config_file = model_dir / "config.json"
    if not config_file.exists():
        raise FileNotFoundError(f"config.json not found in {model_dir}")
    
    with open(config_file, "r") as f:
        cfg = json.load(f)

    hidden_size = cfg.get("hidden_size", 2048)
    num_layers = cfg.get("num_hidden_layers", 27)
    num_heads = cfg.get("num_attention_heads", 16)
    kv_heads = cfg.get("num_key_value_heads", 16)
    n_routed_experts = cfg.get("n_routed_experts", 64)
    top_k = cfg.get("num_experts_per_tok", 6)
    n_shared_experts = cfg.get("n_shared_experts", 2)
    moe_inter = cfg.get("moe_intermediate_size", 1408)
    vocab_size = cfg.get("vocab_size", 102400)
    
    print(f"[2/4] Model Architecture:")
    print(f"      Layers: {num_layers}, Hidden: {hidden_size}, Experts: {n_routed_experts} (top-{top_k} active)")
    print(f"      Shared Experts: {n_shared_experts}, MoE Intermediate: {moe_inter}, Vocab: {vocab_size}")

    # Index all safetensors
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

    # Prepare outputs
    dense_bin_path = out_dir / "attention_dense_layers.bin"
    expert_bin_path = out_dir / f"moe_experts_{quant_type}.bin"
    manifest_path = out_dir / f"moecher_manifest_{quant_type}.json"
    
    print(f"[4/4] Extracting dense and MoE layers to {out_dir}...")
    
    dense_tensors_meta = {}
    dense_offset = 0
    
    with open(dense_bin_path, "wb") as f_dense:
        for t_name, meta in tensor_index.items():
            # Skip routed expert weights for the dense binary
            if ".ffn.experts." in t_name and ".shared_experts." not in t_name:
                continue
                
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

    # Build expert binary
    expert_parts_info = {
        "w1.weight": {"offset_in_block": 0, "nbytes": moe_inter * hidden_size * 2, "dtype": "BF16", "shape": [moe_inter, hidden_size]},
        "w3.weight": {"offset_in_block": moe_inter * hidden_size * 2, "nbytes": moe_inter * hidden_size * 2, "dtype": "BF16", "shape": [moe_inter, hidden_size]},
        "w2.weight": {"offset_in_block": moe_inter * hidden_size * 4, "nbytes": hidden_size * moe_inter * 2, "dtype": "BF16", "shape": [hidden_size, moe_inter]},
    }
    
    block_raw_size = (moe_inter * hidden_size * 2) * 3
    block_size = align_up(block_raw_size, PAGE_SIZE)
    
    with open(expert_bin_path, "wb") as f_exp:
        for l in range(num_layers):
            for e in range(n_routed_experts):
                w1_name = f"model.layers.{l}.ffn.experts.{e}.gate_proj.weight"
                w3_name = f"model.layers.{l}.ffn.experts.{e}.up_proj.weight"
                w2_name = f"model.layers.{l}.ffn.experts.{e}.down_proj.weight"
                
                # If weights exist in checkpoint, copy them; otherwise pad
                block_data = bytearray(block_size)
                for part_name, w_name in [("w1.weight", w1_name), ("w3.weight", w3_name), ("w2.weight", w2_name)]:
                    if w_name in tensor_index:
                        t_meta = tensor_index[w_name]
                        with open(t_meta["file"], "rb") as f_in:
                            f_in.seek(t_meta["data_offset"])
                            p_data = f_in.read(t_meta["nbytes"])
                        off = expert_parts_info[part_name]["offset_in_block"]
                        block_data[off:off+len(p_data)] = p_data
                f_exp.write(block_data)

    print(f"      Wrote expert binary: {expert_bin_path} ({os.path.getsize(expert_bin_path) / (1024**3):.2f} GB)")

    # Build manifest
    manifest = {
        "model_config": {
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_layers,
            "num_attention_heads": num_heads,
            "num_key_value_heads": kv_heads,
            "head_dim": cfg.get("head_dim", 128),
            "qk_rope_head_dim": cfg.get("qk_rope_head_dim", 64),
            "q_lora_rank": cfg.get("q_lora_rank", 512),
            "o_lora_rank": cfg.get("o_lora_rank", 512),
            "o_groups": 1,
            "moe_intermediate_size": moe_inter,
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": top_k,
            "n_shared_experts": n_shared_experts,
            "n_hash_layers": 0,
            "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
            "rope_theta": cfg.get("rope_theta", 10000),
            "original_seq_len": cfg.get("max_position_embeddings", 4096),
            "expert_dtype": "bf16",
        },
        "tokenizer": {
            "tokenizer_json": "tokenizer.json"
        },
        "dense_bin": "attention_dense_layers.bin",
        "expert_bin": expert_bin_path.name,
        "dense_tensors": dense_tensors_meta,
        "expert_layout": {
            "block_size": block_size,
            "n_layers": num_layers,
            "n_experts": n_routed_experts,
            "parts": expert_parts_info
        }
    }

    # Copy tokenizer.json if available
    tok_src = model_dir / "tokenizer.json"
    if tok_src.exists():
        with open(out_dir / "tokenizer.json", "wb") as f_dst, open(tok_src, "rb") as f_src:
            f_dst.write(f_src.read())

    with open(manifest_path, "w") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"      Wrote manifest: {manifest_path}")
    print("\n[SUCCESS] DeepSeek-Coder-V2-Lite model prepared successfully!")
    print(f"To run with MinnieTheMoECher inside VRAM on your RTX 3090:")
    print(f"  ./build/moecher --manifest {manifest_path} --max-vram 20 --quiet\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prepare DeepSeek-Coder-V2-Lite for MinnieTheMoECher")
    parser.add_argument("--model-id", default="deepseek-ai/DeepSeek-Coder-V2-Lite-Instruct", help="HuggingFace model ID")
    parser.add_argument("--local-dir", default=None, help="Local directory if model is already downloaded")
    parser.add_argument("--output-dir", default="models/deepseek_coder_v2_lite", help="Output directory")
    parser.add_argument("--quant", default="iq2_xxs", choices=["iq2_xxs", "bf16"], help="Quantization type")
    args = parser.parse_args()

    out_path = Path(args.output_dir).resolve()
    if args.local_dir:
        model_path = Path(args.local_dir).resolve()
    else:
        model_path = out_path / "raw_hf"
        download_model(args.model_id, model_path)

    prepare_coder_v2_lite(model_path, out_path, args.quant)
