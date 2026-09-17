#!/usr/bin/env python3
"""
quantize_qwen_vision.py — Mixed 3-bit / 4-bit Vision Quantizer for Qwen 3.8 27B
Target: ~13.1 GB footprint (qwen3.8-27B-Vision-13G) to run fully in 16 GB VRAM.

Architecture Summary:
- Visual Tower (model.visual.*): Retained in raw BF16 (~0.92 GB) for full vision/OCR fidelity
- Embeddings (embed_tokens): INT4 block-32 (~0.72 GB instead of 2.54 GB)
- LM Head (lm_head): INT4 block-32 (~0.72 GB instead of 2.54 GB)
- Attention Projections (linear_attn + self_attn): INT4 block-32 (~3.17 GB)
- MLP Feed-Forward (gate_proj, up_proj, down_proj): INT3 block-32 (~7.49 GB instead of 10.16 GB)
- Total Weight Footprint: ~13.1 - 13.5 GB (leaves ~2.5 - 2.9 GB free on 16GB GPUs)

Usage:
    # 1. From an existing MinnieTheMoEcher Qwen manifest + dense bin:
    python3 scripts/quantize_qwen_vision.py \
        --manifest-in models/qwen3_8_27b/moecher_manifest_qwen.json \
        --dense-bin-in models/qwen3_8_27b/attention_dense_layers.bin \
        --output-dir models/qwen3.8-27B-Vision-13G

    # 2. Or from raw Hugging Face safetensors:
    python3 scripts/quantize_qwen_vision.py \
        --input-dir /path/to/raw_hf \
        --output-dir models/qwen3.8-27B-Vision-13G
"""

import os
import sys
import json
import time
import shutil
import struct
import argparse
from pathlib import Path
from typing import Dict, List, Tuple, Any

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import numpy as np

# ── INT4 Block-32 Quantizer ───────────────────────────────────────────────────

def quantize_matrix_int4_block32(tensor_f32: np.ndarray, block_size: int = 32) -> Tuple[bytes, bytes]:
    """
    Quantizes 2D float32 array [rows, cols] into symmetric INT4 with block size 32.
    Returns:
        packed_bytes: bytes (uint8 packed, shape [rows, cols // 2])
        scale_bytes: bytes (bfloat16 raw bytes, shape [rows, cols // block_size])
    """
    rows, cols = tensor_f32.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by block_size {block_size}"
    num_blocks = cols // block_size

    # Guard against NaNs or Infs
    tensor_f32 = np.nan_to_num(tensor_f32, nan=0.0, posinf=0.0, neginf=0.0)

    # Reshape into [rows, num_blocks, block_size]
    t_blocks = tensor_f32.reshape(rows, num_blocks, block_size)

    # Scale per block: max_abs / 7.0
    max_abs = np.max(np.abs(t_blocks), axis=-1, keepdims=True)
    scale = np.maximum(max_abs / 7.0, 1e-8)

    # Quantize to [-8, 7]
    q = np.clip(np.round(t_blocks / scale), -8, 7).astype(np.int32)

    # Offset by +8 to get unsigned [0, 15]
    u = (q + 8).astype(np.uint8).reshape(rows, cols)

    # Pack 2 values per byte (even column in low nibble, odd column in high nibble)
    u_even = u[:, 0::2]
    u_odd = u[:, 1::2]
    packed = (u_even | (u_odd << 4)).tobytes()

    # Scale in BF16 bytes
    scale_f32 = scale.squeeze(-1).astype(np.float32)
    u32 = scale_f32.view(np.uint32)
    scale_bf16 = (u32 >> 16).astype(np.uint16)
    scale_bytes = scale_bf16.tobytes()

    return packed, scale_bytes


# ── INT3 Block-32 Quantizer (Vectorized) ──────────────────────────────────────

def quantize_matrix_int3_block32(tensor_f32: np.ndarray, block_size: int = 32) -> Tuple[bytes, bytes]:
    """
    Quantizes 2D float32 array [rows, cols] into symmetric INT3 with block size 32.
    INT3 values are in [-4, 3], offset by +4 to [0, 7] (3 bits).
    8 values pack into exactly 3 bytes (24 bits):
      Byte 0: v0[0..2] | (v1[0..2] << 3) | ((v2 & 0x3) << 6)
      Byte 1: ((v2 >> 2) & 0x1) | (v3[0..2] << 1) | (v4[0..2] << 4) | ((v5 & 0x1) << 7)
      Byte 2: ((v5 >> 1) & 0x3) | (v6[0..2] << 2) | (v7[0..2] << 5)
    32 values pack into 12 bytes.
    Scale per block: max_abs / 3.0 (stored as bfloat16, 2 bytes).
    Total per 32 weights: 14 bytes (3.50 bpw).
    """
    rows, cols = tensor_f32.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by block_size {block_size}"
    assert block_size % 8 == 0, f"block_size {block_size} must be divisible by 8"
    num_blocks = cols // block_size

    # Guard against NaNs or Infs
    tensor_f32 = np.nan_to_num(tensor_f32, nan=0.0, posinf=0.0, neginf=0.0)

    # Reshape into [rows, num_blocks, block_size]
    t_blocks = tensor_f32.reshape(rows, num_blocks, block_size)

    # Scale per block: max_abs / 3.0
    max_abs = np.max(np.abs(t_blocks), axis=-1, keepdims=True)
    scale = np.maximum(max_abs / 3.0, 1e-8)

    # Quantize to [-4, 3]
    q = np.clip(np.round(t_blocks / scale), -4, 3).astype(np.int32)

    # Offset by +4 to get unsigned [0, 7]
    u = (q + 4).astype(np.uint8).reshape(rows, cols)

    # Vectorized packing: 8 values per 3 bytes
    v = u.reshape(rows, cols // 8, 8)
    v0, v1, v2, v3 = v[:, :, 0], v[:, :, 1], v[:, :, 2], v[:, :, 3]
    v4, v5, v6, v7 = v[:, :, 4], v[:, :, 5], v[:, :, 6], v[:, :, 7]

    b0 = (v0 & 0x7) | ((v1 & 0x7) << 3) | ((v2 & 0x3) << 6)
    b1 = ((v2 >> 2) & 0x1) | ((v3 & 0x7) << 1) | ((v4 & 0x7) << 4) | ((v5 & 0x1) << 7)
    b2 = ((v5 >> 1) & 0x3) | ((v6 & 0x7) << 2) | ((v7 & 0x7) << 5)

    packed = np.stack([b0, b1, b2], axis=-1).reshape(rows, (cols * 3) // 8).tobytes()

    # Scale in BF16 bytes
    scale_f32 = scale.squeeze(-1).astype(np.float32)
    u32 = scale_f32.view(np.uint32)
    scale_bf16 = (u32 >> 16).astype(np.uint16)
    scale_bytes = scale_bf16.tobytes()

    return packed, scale_bytes


# ── Dequantization Helpers (for reading existing INT4 dense binary) ────────────

def dequantize_int4_block32(packed_u8: np.ndarray, scale_bf16: np.ndarray, shape: list, block_size: int = 32) -> np.ndarray:
    """Dequantizes INT4 block-32 packed bytes and BF16 scales back to float32 matrix."""
    rows, cols = shape
    packed = packed_u8.reshape(rows, cols // 2)

    u_even = packed & 0x0F
    u_odd = (packed >> 4) & 0x0F
    u = np.empty((rows, cols), dtype=np.uint8)
    u[:, 0::2] = u_even
    u[:, 1::2] = u_odd

    q = u.astype(np.int32) - 8

    # Decode BF16 scale to float32
    s_u16 = scale_bf16.reshape(rows, cols // block_size)
    s_f32 = (s_u16.astype(np.uint32) << 16).view(np.float32)

    # Broadcast scale across blocks
    q_blocks = q.reshape(rows, cols // block_size, block_size)
    w_blocks = q_blocks.astype(np.float32) * s_f32[:, :, np.newaxis]
    return w_blocks.reshape(rows, cols)


# ── Safetensors Header Parser ─────────────────────────────────────────────────

def read_safetensor_header(path: str):
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json.decode("utf-8"))
    return header, 8 + header_size


# ── Quantizer Core ────────────────────────────────────────────────────────────

def quantize_qwen_vision(
    manifest_in: Path = None,
    dense_bin_in: Path = None,
    raw_hf_dir: Path = None,
    output_dir: Path = None,
    mlp_bits: int = 3,
    mlp_scheme: str = "all-3bit",
    visual_mode: str = "bf16",
    quantize_embed: bool = True,
    quantize_head: bool = True,
    block_size: int = 32
):
    output_dir.mkdir(parents=True, exist_ok=True)
    out_bin_path = output_dir / "attention_dense_layers.bin"
    out_manifest_path = output_dir / "moecher_manifest.json"

    print("═" * 70)
    print("  Qwen 3.8 27B Vision Quantizer -> Target: ~13.1 GB (16GB VRAM Target)")
    print(f"  MLP Precision     : {mlp_bits}-bit ({mlp_scheme})")
    print(f"  Visual Tower      : {visual_mode.upper()} (preserved)")
    print(f"  Embeddings / Head : {'INT4' if quantize_embed and quantize_head else 'BF16'}")
    print(f"  Target Output     : {output_dir}")
    print("═" * 70)

    # Mode 1: Convert from existing MinnieTheMoEcher manifest + dense bin
    if manifest_in and dense_bin_in and manifest_in.exists() and dense_bin_in.exists():
        print(f"\n[Source Mode] Reading existing dense binary: {dense_bin_in}")
        print(f"              Manifest: {manifest_in}")
        with open(manifest_in, "r") as f:
            base_manifest = json.load(f)

        dense_tensors = base_manifest["dense_tensors"]
        total_tensors = len(dense_tensors)
        dense_tensors_meta = {}
        dense_offset = 0

        total_orig_bytes = os.path.getsize(dense_bin_in)
        total_quant_bytes = 0
        t0 = time.time()

        # Build map of scale tensors to their parent weight
        scale_to_weight = {}
        for t_name, meta in dense_tensors.items():
            if "scale_offset" in meta:
                # In MoEcher format, scale is stored inline in the same entry or separate
                pass

        with open(dense_bin_in, "rb") as f_in, open(out_bin_path, "wb") as f_out:
            for idx, (t_name, meta) in enumerate(dense_tensors.items()):
                shape = meta.get("shape", [])
                dtype = meta.get("dtype", "BF16")
                offset = meta.get("offset", 0)
                nbytes = meta.get("nbytes", 0)

                f_in.seek(offset)
                raw_data = f_in.read(nbytes)

                is_2d = (len(shape) == 2 and shape[0] >= 512 and shape[1] >= 512 and shape[1] % block_size == 0)
                is_visual = ("visual" in t_name)
                is_embed = ("embed" in t_name)
                is_head = ("lm_head" in t_name or "head.weight" in t_name)
                is_mlp = ("mlp" in t_name)
                is_attn = ("attn" in t_name and not is_visual)

                # Determine target quantization format for this tensor
                target_format = "BF16"

                if is_visual:
                    if visual_mode == "int4" and is_2d:
                        target_format = "int4"
                    else:
                        target_format = "BF16"
                elif is_embed and quantize_embed and is_2d:
                    target_format = "int4"
                elif is_head and quantize_head and is_2d:
                    target_format = "int4"
                elif is_mlp and is_2d:
                    if mlp_scheme == "hybrid" and "down_proj" in t_name:
                        target_format = "int4"
                    else:
                        target_format = f"int{mlp_bits}"
                elif is_attn and is_2d:
                    target_format = "int4"
                else:
                    target_format = dtype # Preserve original

                # Check if tensor already matches target
                if dtype == target_format and target_format == "int4":
                    # Already INT4 in source binary! Copy packed weight and scale directly
                    w_offset = dense_offset
                    f_out.write(raw_data)
                    dense_offset += nbytes

                    # Read scale
                    scale_offset = dense_offset
                    scale_nbytes = meta.get("scale_nbytes", 0)
                    f_in.seek(meta["scale_offset"])
                    scale_raw = f_in.read(scale_nbytes)
                    f_out.write(scale_raw)
                    dense_offset += scale_nbytes

                    total_quant_bytes += (nbytes + scale_nbytes)
                    dense_tensors_meta[t_name] = {
                        "offset": w_offset,
                        "nbytes": nbytes,
                        "dtype": "int4",
                        "shape": shape,
                        "scale_offset": scale_offset,
                        "scale_nbytes": scale_nbytes,
                        "scale_dtype": meta.get("scale_dtype", "bfloat16"),
                        "block_size": meta.get("block_size", block_size)
                    }
                    continue

                # If conversion is needed:
                # 1. Recover float32 weight matrix
                if dtype == "int4":
                    f_in.seek(meta["scale_offset"])
                    scale_raw = f_in.read(meta["scale_nbytes"])
                    packed_u8 = np.frombuffer(raw_data, dtype=np.uint8)
                    scale_bf16 = np.frombuffer(scale_raw, dtype=np.uint16)
                    w_f32 = dequantize_int4_block32(packed_u8, scale_bf16, shape, block_size)
                elif dtype == "BF16":
                    u16 = np.frombuffer(raw_data, dtype=np.uint16)
                    w_f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)
                else:
                    w_f32 = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)

                # 2. Quantize according to target format
                if target_format == "int4":
                    packed_w, packed_scale = quantize_matrix_int4_block32(w_f32, block_size)
                    w_offset = dense_offset
                    w_nbytes = len(packed_w)
                    f_out.write(packed_w)
                    dense_offset += w_nbytes

                    scale_offset = dense_offset
                    scale_nbytes = len(packed_scale)
                    f_out.write(packed_scale)
                    dense_offset += scale_nbytes

                    total_quant_bytes += (w_nbytes + scale_nbytes)
                    dense_tensors_meta[t_name] = {
                        "offset": w_offset,
                        "nbytes": w_nbytes,
                        "dtype": "int4",
                        "shape": shape,
                        "scale_offset": scale_offset,
                        "scale_nbytes": scale_nbytes,
                        "scale_dtype": "bfloat16",
                        "block_size": block_size
                    }
                elif target_format == "int3":
                    packed_w, packed_scale = quantize_matrix_int3_block32(w_f32, block_size)
                    w_offset = dense_offset
                    w_nbytes = len(packed_w)
                    f_out.write(packed_w)
                    dense_offset += w_nbytes

                    scale_offset = dense_offset
                    scale_nbytes = len(packed_scale)
                    f_out.write(packed_scale)
                    dense_offset += scale_nbytes

                    total_quant_bytes += (w_nbytes + scale_nbytes)
                    dense_tensors_meta[t_name] = {
                        "offset": w_offset,
                        "nbytes": w_nbytes,
                        "dtype": "int3",
                        "shape": shape,
                        "scale_offset": scale_offset,
                        "scale_nbytes": scale_nbytes,
                        "scale_dtype": "bfloat16",
                        "block_size": block_size
                    }
                else:
                    # Stored in BF16 / original precision
                    f_out.write(raw_data)
                    dense_tensors_meta[t_name] = {
                        "offset": dense_offset,
                        "nbytes": nbytes,
                        "dtype": dtype,
                        "shape": shape
                    }
                    dense_offset += nbytes
                    total_quant_bytes += nbytes

                if idx % 30 == 0 or idx == total_tensors - 1:
                    cur_size = dense_tensors_meta[t_name]["nbytes"] + dense_tensors_meta[t_name].get("scale_nbytes", 0)
                    print(f"[{idx+1}/{total_tensors}] {target_format.upper():5s}: {t_name} {shape} -> {cur_size / (1024**2):.2f} MB")

        # Copy tokenizer if available
        src_tok = manifest_in.parent / "tokenizer.json"
        if src_tok.exists():
            shutil.copy2(src_tok, output_dir / "tokenizer.json")

    # Mode 2: Convert from raw safetensors directory
    elif raw_hf_dir and raw_hf_dir.exists():
        print(f"\n[Source Mode] Reading raw safetensors from: {raw_hf_dir}")
        st_files = sorted(list(raw_hf_dir.glob("*.safetensors")))
        if not st_files:
            raise FileNotFoundError(f"No safetensors found in {raw_hf_dir}")

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
                    "dtype": info["dtype"]
                }

        total_tensors = len(tensor_index)
        dense_tensors_meta = {}
        dense_offset = 0
        total_orig_bytes = sum(m["nbytes"] for m in tensor_index.values())
        total_quant_bytes = 0
        t0 = time.time()

        with open(out_bin_path, "wb") as f_out:
            for idx, (t_name, meta) in enumerate(tensor_index.items()):
                shape = meta["shape"]
                dtype = meta["dtype"]
                nbytes = meta["nbytes"]

                with open(meta["file"], "rb") as f_in:
                    f_in.seek(meta["data_offset"])
                    raw_data = f_in.read(nbytes)

                is_2d = (len(shape) == 2 and shape[0] >= 512 and shape[1] >= 512 and shape[1] % block_size == 0)
                is_visual = ("visual" in t_name)
                is_embed = ("embed" in t_name)
                is_head = ("lm_head" in t_name or "head.weight" in t_name)
                is_mlp = ("mlp" in t_name)
                is_attn = ("attn" in t_name and not is_visual)

                target_format = "BF16"
                if is_visual:
                    target_format = "int4" if (visual_mode == "int4" and is_2d) else "BF16"
                elif is_embed and quantize_embed and is_2d:
                    target_format = "int4"
                elif is_head and quantize_head and is_2d:
                    target_format = "int4"
                elif is_mlp and is_2d:
                    target_format = "int4" if (mlp_scheme == "hybrid" and "down_proj" in t_name) else f"int{mlp_bits}"
                elif is_attn and is_2d:
                    target_format = "int4"
                else:
                    target_format = dtype

                if target_format in ["int4", "int3"]:
                    # Convert BF16 to float32
                    u16 = np.frombuffer(raw_data, dtype=np.uint16)
                    w_f32 = (u16.astype(np.uint32) << 16).view(np.float32).reshape(shape)

                    if target_format == "int4":
                        packed_w, packed_scale = quantize_matrix_int4_block32(w_f32, block_size)
                    else:
                        packed_w, packed_scale = quantize_matrix_int3_block32(w_f32, block_size)

                    w_offset = dense_offset
                    w_nbytes = len(packed_w)
                    f_out.write(packed_w)
                    dense_offset += w_nbytes

                    scale_offset = dense_offset
                    scale_nbytes = len(packed_scale)
                    f_out.write(packed_scale)
                    dense_offset += scale_nbytes

                    total_quant_bytes += (w_nbytes + scale_nbytes)
                    dense_tensors_meta[t_name] = {
                        "offset": w_offset,
                        "nbytes": w_nbytes,
                        "dtype": target_format,
                        "shape": shape,
                        "scale_offset": scale_offset,
                        "scale_nbytes": scale_nbytes,
                        "scale_dtype": "bfloat16",
                        "block_size": block_size
                    }
                else:
                    f_out.write(raw_data)
                    dense_tensors_meta[t_name] = {
                        "offset": dense_offset,
                        "nbytes": nbytes,
                        "dtype": dtype,
                        "shape": shape
                    }
                    dense_offset += nbytes
                    total_quant_bytes += nbytes

                if idx % 30 == 0 or idx == total_tensors - 1:
                    cur_size = dense_tensors_meta[t_name]["nbytes"] + dense_tensors_meta[t_name].get("scale_nbytes", 0)
                    print(f"[{idx+1}/{total_tensors}] {target_format.upper():5s}: {t_name} {shape} -> {cur_size / (1024**2):.2f} MB")

        src_tok = raw_hf_dir / "tokenizer.json"
        if src_tok.exists():
            shutil.copy2(src_tok, output_dir / "tokenizer.json")
    else:
        raise ValueError("Must provide either (--manifest-in and --dense-bin-in) or --input-dir")

    # Build and write final manifest
    cfg = base_manifest.get("model_config", {}) if 'base_manifest' in locals() else {}
    manifest = {
        "model_config": {
            "model_id": "qwen3.8-27B-Vision-13G",
            "model_name": "Qwen 3.8 27B Vision (13G Mixed Quant)",
            "architecture": "qwen2",
            "has_vision": True,
            "vocab_size": cfg.get("vocab_size", 248320),
            "hidden_size": cfg.get("hidden_size", 5120),
            "num_hidden_layers": cfg.get("num_hidden_layers", 64),
            "num_attention_heads": cfg.get("num_attention_heads", 24),
            "num_key_value_heads": cfg.get("num_key_value_heads", 4),
            "head_dim": cfg.get("head_dim", 256),
            "intermediate_size": cfg.get("intermediate_size", 17408),
            "rms_norm_eps": cfg.get("rms_norm_eps", 1e-6),
            "rope_theta": cfg.get("rope_theta", 10000000.0),
            "max_seq_len": cfg.get("max_seq_len", 32768),
            "original_seq_len": cfg.get("original_seq_len", 262144),
            "eos_token_id": cfg.get("eos_token_id", 248046),
            "expert_dtype": "none",
            "n_routed_experts": 0,
            "num_experts_per_tok": 0,
            "n_shared_experts": 0,
            "n_hash_layers": 0
        },
        "visual_config": {
            "depth": 32,
            "embed_dim": 1152,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2
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

    with open(out_manifest_path, "w") as f_man:
        json.dump(manifest, f_man, indent=2)

    elapsed = time.time() - t0
    final_file_size = os.path.getsize(out_bin_path)

    print("\n" + "═" * 70)
    print("  Quantization Complete!")
    print(f"  Time Elapsed      : {elapsed:.1f} seconds")
    print(f"  Original Binary   : {total_orig_bytes / (1024**3):.2f} GiB ({total_orig_bytes / 1e9:.2f} GB)")
    print(f"  Quantized Binary  : {final_file_size / (1024**3):.2f} GiB ({final_file_size / 1e9:.2f} GB)")
    print(f"  Saved VRAM        : {(total_orig_bytes - final_file_size) / 1e9:.2f} GB")
    print(f"  Compression Ratio : {total_orig_bytes / final_file_size:.2f}x")
    print(f"  Free VRAM on 16GB : {16.0 - (final_file_size / 1e9):.2f} GB (Ready for Windows/Linux full VRAM residency!)")
    print(f"  Manifest written  : {out_manifest_path}")
    print("═" * 70)


# ── CLI Interface ─────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Quantize Qwen 3.8 27B into qwen3.8-27B-Vision-13G (~13.1 GB) for 16GB GPUs"
    )
    parser.add_argument("--manifest-in", type=Path, default=None,
                        help="Path to source moecher_manifest.json")
    parser.add_argument("--dense-bin-in", type=Path, default=None,
                        help="Path to source attention_dense_layers.bin")
    parser.add_argument("--input-dir", type=Path, default=None,
                        help="Directory containing source model or raw safetensors")
    parser.add_argument("--output-dir", type=Path, default=Path("models/qwen3.8-27B-Vision-13G"),
                        help="Directory to save quantized binary and manifest")
    parser.add_argument("--mlp-bits", type=int, default=3, choices=[3, 4],
                        help="Bitwidth for MLP layers (default: 3 for ~13.1 GB target)")
    parser.add_argument("--mlp-scheme", type=str, default="all-3bit", choices=["all-3bit", "hybrid"],
                        help="MLP quantization scheme: all-3bit or hybrid (gate/up 3-bit, down 4-bit)")
    parser.add_argument("--visual-mode", type=str, default="bf16", choices=["bf16", "int4"],
                        help="Precision for visual encoder: bf16 (full fidelity) or int4 (maximum compression)")
    parser.add_argument("--no-embed-quant", action="store_true",
                        help="Leave embed_tokens in BF16 (adds +1.82 GB)")
    parser.add_argument("--no-head-quant", action="store_true",
                        help="Leave lm_head in BF16 (adds +1.82 GB)")
    parser.add_argument("--block-size", type=int, default=32,
                        help="Quantization block size (default: 32)")

    args = parser.parse_args()

    # Automatically resolve paths if --input-dir is passed
    manifest_in = args.manifest_in
    dense_bin_in = args.dense_bin_in
    raw_hf_dir = None

    if args.input_dir:
        input_path = args.input_dir
        if (input_path / "moecher_manifest.json").exists():
            manifest_in = input_path / "moecher_manifest.json"
        elif (input_path / "moecher_manifest_qwen.json").exists():
            manifest_in = input_path / "moecher_manifest_qwen.json"
        elif (input_path / "moecher_manifest_qwen_q4.json").exists():
            manifest_in = input_path / "moecher_manifest_qwen_q4.json"

        if (input_path / "attention_dense_layers.bin").exists():
            dense_bin_in = input_path / "attention_dense_layers.bin"
        elif (input_path / "attention_dense_layers_q4.bin").exists():
            dense_bin_in = input_path / "attention_dense_layers_q4.bin"

        if (input_path / "raw_hf").exists() and any((input_path / "raw_hf").glob("*.safetensors")):
            raw_hf_dir = input_path / "raw_hf"
        elif any(input_path.glob("*.safetensors")):
            raw_hf_dir = input_path

    # Automatically resolve dense_bin_in from manifest_in if not explicitly specified
    if manifest_in and not dense_bin_in:
        parent_dir = manifest_in.parent
        if manifest_in.exists():
            try:
                with open(manifest_in, "r") as f:
                    man_data = json.load(f)
                if "dense_bin" in man_data:
                    cand = parent_dir / man_data["dense_bin"]
                    if cand.exists():
                        dense_bin_in = cand
            except Exception:
                pass
        if not dense_bin_in:
            for cand_name in ["attention_dense_layers_q4.bin", "attention_dense_layers.bin"]:
                if (parent_dir / cand_name).exists():
                    dense_bin_in = parent_dir / cand_name
                    break

    # Default fallback: check models/qwen3_8_27b_q4
    if not manifest_in and not raw_hf_dir:
        fallback_dir = Path("models/qwen3_8_27b_q4")
        if (fallback_dir / "moecher_manifest.json").exists():
            manifest_in = fallback_dir / "moecher_manifest.json"
            if (fallback_dir / "attention_dense_layers_q4.bin").exists():
                dense_bin_in = fallback_dir / "attention_dense_layers_q4.bin"
            elif (fallback_dir / "attention_dense_layers.bin").exists():
                dense_bin_in = fallback_dir / "attention_dense_layers.bin"

    quantize_qwen_vision(
        manifest_in=manifest_in,
        dense_bin_in=dense_bin_in,
        raw_hf_dir=raw_hf_dir,
        output_dir=args.output_dir,
        mlp_bits=args.mlp_bits,
        mlp_scheme=args.mlp_scheme,
        visual_mode=args.visual_mode,
        quantize_embed=(not args.no_embed_quant),
        quantize_head=(not args.no_head_quant),
        block_size=args.block_size
    )


if __name__ == "__main__":
    main()
