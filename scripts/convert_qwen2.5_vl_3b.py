#!/usr/bin/env python3
"""
convert_qwen2.5_vl_3b.py
Convert and quantize Qwen2.5-VL-3B-Instruct to MinnieTheMoECher binary format:
- LLM dense layers quantized to INT4 block-32 (scales in BF16)
- Embeddings and RMSNorms in BF16
- ViT Vision Tower (32 blocks, embed_dim 1280) in BF16
- Produces models/frankenstin/vision/attention_dense_layers.bin
  and models/frankenstin/vision/moecher_manifest.json
"""

import os
import sys
import json
import time
import struct
import shutil
from pathlib import Path

import torch
import numpy as np
from safetensors import safe_open

PAGE_SIZE = 4096

def quantize_matrix_int4_block32(tensor: torch.Tensor, block_size: int = 32):
    rows, cols = tensor.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by {block_size}"
    num_blocks = cols // block_size

    t = tensor.to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    t_blocks = t.view(rows, num_blocks, block_size)

    max_abs = torch.max(torch.abs(t_blocks), dim=-1, keepdim=True).values
    scale = torch.clamp(max_abs / 7.0, min=1e-8)

    q = torch.clamp(torch.round(t_blocks / scale), -8, 7).to(torch.int32)
    u = (q + 8).to(torch.uint8).view(rows, cols)

    u_even = u[:, 0::2]
    u_odd = u[:, 1::2]
    packed = u_even | (u_odd << 4)

    scale_bf16 = scale.squeeze(-1).to(torch.bfloat16)
    packed_bytes = packed.cpu().numpy().tobytes()
    scale_bytes = scale_bf16.view(torch.int16).cpu().numpy().tobytes()

    return packed_bytes, scale_bytes

def main():
    snap_dir = Path("/home/tinobruno/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3")
    out_dir = Path("models/frankenstin/vision")
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Converting Qwen2.5-VL-3B-Instruct to {out_dir} ===")
    start_time = time.time()

    # 1. Load config
    with open(snap_dir / "config.json") as f:
        cfg = json.load(f)

    # Copy tokenizer and processor configs
    for fname in ["tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "preprocessor_config.json", "chat_template.json"]:
        src = snap_dir / fname
        if src.exists():
            shutil.copy2(src, out_dir / fname)
            print(f"Copied {fname} to {out_dir}")

    # 2. Index safetensors
    st_files = sorted(list(snap_dir.glob("*.safetensors")))
    print(f"Indexing {len(st_files)} safetensors shards...")
    tensor_map = {}
    for st_path in st_files:
        with safe_open(str(st_path), framework="pt") as f:
            for k in f.keys():
                tensor_map[k] = str(st_path)

    dense_bin_path = out_dir / "attention_dense_layers.bin"
    manifest_path = out_dir / "moecher_manifest.json"

    dense_meta = {}
    curr_offset = 0

    print(f"Writing packed binary to {dense_bin_path}...")
    with open(dense_bin_path, "wb") as fout:
        def write_bytes(data: bytes, name: str, shape: list, dtype: str, original_shape: list = None):
            nonlocal curr_offset
            # Align to 64 bytes
            align = (64 - (curr_offset % 64)) % 64
            if align > 0:
                fout.write(b"\x00" * align)
                curr_offset += align

            offset = curr_offset
            fout.write(data)
            nbytes = len(data)
            curr_offset += nbytes

            meta_entry = {
                "offset": offset,
                "nbytes": nbytes,
                "dtype": dtype,
                "shape": shape
            }
            if original_shape:
                meta_entry["original_shape"] = original_shape
            dense_meta[name] = meta_entry
            return offset

        # Helper to load a tensor
        def get_tensor(name: str) -> torch.Tensor:
            st_path = tensor_map[name]
            with safe_open(st_path, framework="pt") as f:
                return f.get_tensor(name)

        # 3. Embeddings & Final Norm
        print("Processing embeddings and final norm...")
        embed_t = get_tensor("model.embed_tokens.weight").to(torch.bfloat16)
        write_bytes(embed_t.view(torch.int16).cpu().numpy().tobytes(),
                    "model.embed_tokens.weight", list(embed_t.shape), "BF16")

        norm_t = get_tensor("model.norm.weight").to(torch.bfloat16)
        write_bytes(norm_t.view(torch.int16).cpu().numpy().tobytes(),
                    "model.norm.weight", list(norm_t.shape), "BF16")

        # 4. Language Model Layers (36 layers)
        num_layers = cfg.get("num_hidden_layers", 36)
        print(f"Processing {num_layers} language model layers (INT4 block-32)...")
        for layer_idx in range(num_layers):
            pfx = f"model.layers.{layer_idx}."
            if layer_idx % 6 == 0 or layer_idx == num_layers - 1:
                print(f"  Layer {layer_idx}/{num_layers}...")

            # Input Layernorm (BF16)
            in_norm = get_tensor(pfx + "input_layernorm.weight").to(torch.bfloat16)
            write_bytes(in_norm.view(torch.int16).cpu().numpy().tobytes(),
                        pfx + "input_layernorm.weight", list(in_norm.shape), "BF16")

            # Post Attention Layernorm (BF16)
            post_norm = get_tensor(pfx + "post_attention_layernorm.weight").to(torch.bfloat16)
            write_bytes(post_norm.view(torch.int16).cpu().numpy().tobytes(),
                        pfx + "post_attention_layernorm.weight", list(post_norm.shape), "BF16")

            # Attention projections (INT4)
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj"]:
                w = get_tensor(pfx + proj + ".weight")
                packed, scale = quantize_matrix_int4_block32(w, block_size=32)
                write_bytes(packed, pfx + proj + ".weight", list(w.shape), "int4")
                write_bytes(scale, pfx + proj + ".weight_scale", [w.shape[0], w.shape[1] // 32], "BF16")
                bias_key = pfx + proj + ".bias"
                if bias_key in tensor_map:
                    bias = get_tensor(bias_key).to(torch.bfloat16)
                    write_bytes(bias.view(torch.int16).cpu().numpy().tobytes(), bias_key, list(bias.shape), "BF16")

            # MLP projections (INT4)
            for proj in ["mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"]:
                w = get_tensor(pfx + proj + ".weight")
                packed, scale = quantize_matrix_int4_block32(w, block_size=32)
                write_bytes(packed, pfx + proj + ".weight", list(w.shape), "int4")
                write_bytes(scale, pfx + proj + ".weight_scale", [w.shape[0], w.shape[1] // 32], "BF16")

        # 5. ViT Vision Tower
        print("Processing ViT Vision Tower tensors...")
        vis_keys = [k for k in tensor_map.keys() if k.startswith("visual.")]
        print(f"Found {len(vis_keys)} visual tensors.")
        for k in sorted(vis_keys):
            t = get_tensor(k).to(torch.bfloat16)
            # Map name to moecher standard: model.visual.*
            m_name = "model." + k if not k.startswith("model.") else k
            write_bytes(t.view(torch.int16).cpu().numpy().tobytes(),
                        m_name, list(t.shape), "BF16")

    total_mb = curr_offset / (1024 * 1024)
    print(f"Wrote {len(dense_meta)} tensors, total size: {total_mb:.2f} MB ({total_mb/1024:.2f} GB)")

    # 6. Generate Manifest
    manifest = {
        "model_config": {
            "architecture": "qwen2",
            "model_name": "Qwen2.5-VL-3B-Instruct",
            "model_id": "qwen2.5-vl-3b-instruct",
            "vocab_size": cfg["vocab_size"],
            "hidden_size": cfg["hidden_size"],
            "num_hidden_layers": cfg["num_hidden_layers"],
            "num_attention_heads": cfg["num_attention_heads"],
            "num_key_value_heads": cfg["num_key_value_heads"],
            "head_dim": cfg["hidden_size"] // cfg["num_attention_heads"],
            "intermediate_size": cfg["intermediate_size"],
            "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
            "rope_theta": cfg.get("rope_theta", 1000000.0),
            "max_seq_len": 8192,
            "bos_token_id": cfg.get("bos_token_id", 151643),
            "eos_token_id": cfg.get("eos_token_id", 151645),
            "has_vision": True,
            "vision_config": cfg.get("vision_config", {})
        },
        "tokenizer": {
            "tokenizer_json": "tokenizer.json"
        },
        "dense_bin": "attention_dense_layers.bin",
        "dense_tensors": dense_meta
    }

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    elapsed = time.time() - start_time
    print(f"Saved manifest to {manifest_path}")
    print(f"=== Conversion completed successfully in {elapsed:.1f}s ===")

if __name__ == "__main__":
    main()
