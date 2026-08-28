#!/usr/bin/env python3
"""
quantize_deepseek_dense.py — Consolidated INT4 Dense Quantizer for DeepSeek V4 Flash
Quantizes MLA attention projections, shared resident experts, and embeddings to INT4 block-32
to reduce dense memory footprint from ~8.8 GB to ~3.9 GB for 8 GB GPUs.

Usage:
    python scripts/quantize_deepseek_dense.py \
        --manifest-in models/deepseek_v4_flash_iq2/moecher_manifest.json \
        --dense-bin-in models/deepseek_v4_flash_iq2/attention_dense_layers.bin \
        --output-dir models/deepseek_v4_flash_q4 \
        --block-size 32
"""

import os
import sys
import json
import time
import argparse
from pathlib import Path

import torch
import numpy as np

# ── FP8 E4M3 + E8M0 Dequantization Helpers ─────────────────────────────────────

def fp8_e4m3_to_float(val_u8: np.ndarray) -> np.ndarray:
    """Decodes uint8 array of FP8 E4M3 numbers to float32."""
    sign = np.where((val_u8 & 0x80) != 0, -1.0, 1.0).astype(np.float32)
    exp = ((val_u8 & 0x78) >> 3).astype(np.float32)
    mant = (val_u8 & 0x07).astype(np.float32)

    val = np.zeros_like(val_u8, dtype=np.float32)
    mask_zero = (val_u8 == 0)
    mask_subnorm = (exp == 0) & (~mask_zero)
    mask_norm = (exp > 0)

    val[mask_norm] = sign[mask_norm] * (2.0 ** (exp[mask_norm] - 7.0)) * (1.0 + mant[mask_norm] / 8.0)
    val[mask_subnorm] = sign[mask_subnorm] * (2.0 ** -6.0) * (mant[mask_subnorm] / 8.0)
    return val

def e8m0_to_float(scale_u8: np.ndarray) -> np.ndarray:
    """Decodes uint8 array of E8M0 exponents to float32 scale factors."""
    return (2.0 ** (scale_u8.astype(np.float32) - 127.0))

def dequantize_fp8_e4m3_matrix(weight_u8: np.ndarray, scale_u8: np.ndarray, shape: list, block_size: int = 128) -> np.ndarray:
    """Dequantizes [rows, cols] FP8 E4M3 matrix with [ceil(rows/128), ceil(cols/128)] E8M0 block scales."""
    rows, cols = shape
    w_f32 = fp8_e4m3_to_float(weight_u8.reshape(rows, cols))
    s_f32 = e8m0_to_float(scale_u8)

    # Apply block scales
    n_block_r = (rows + block_size - 1) // block_size
    n_block_c = (cols + block_size - 1) // block_size
    s_f32 = s_f32.reshape(n_block_r, n_block_c)

    # Broadcast scales across blocks
    for br in range(n_block_r):
        r_start = br * block_size
        r_end = min((br + 1) * block_size, rows)
        for bc in range(n_block_c):
            c_start = bc * block_size
            c_end = min((bc + 1) * block_size, cols)
            w_f32[r_start:r_end, c_start:c_end] *= s_f32[br, bc]

    return w_f32

# ── INT4 Block-32 Quantizer ───────────────────────────────────────────────────

def quantize_matrix_int4_block32(tensor_f32: np.ndarray, block_size: int = 32):
    """
    Quantizes 2D float32 array [rows, cols] into symmetric INT4 with block size 32.
    Returns:
        packed_bytes: bytes (uint8 packed, shape [rows, cols // 2])
        scale_bytes: bytes (bfloat16 raw bytes, shape [rows, cols // block_size])
    """
    rows, cols = tensor_f32.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by block_size {block_size}"
    num_blocks = cols // block_size

    # Reshape into [rows, num_blocks, block_size]
    t_blocks = tensor_f32.reshape(rows, num_blocks, block_size)

    # Scale per block: max_abs / 7.0
    max_abs = np.max(np.abs(t_blocks), axis=-1, keepdims=True)
    scale = np.maximum(max_abs / 7.0, 1e-8)

    # Quantize to [-8, 7]
    q = np.clip(np.round(t_blocks / scale), -8, 7).astype(np.int32)

    # Offset by +8 to get unsigned [0, 15]
    u = (q + 8).astype(np.uint8).reshape(rows, cols)

    # Pack 2 values per byte (even in low nibble, odd in high nibble)
    u_even = u[:, 0::2]
    u_odd = u[:, 1::2]
    packed = (u_even | (u_odd << 4)).tobytes()

    # Scale in BF16 bytes
    scale_f32 = scale.squeeze(-1).astype(np.float32)
    scale_bf16 = torch.from_numpy(scale_f32).to(torch.bfloat16)
    scale_bytes = scale_bf16.view(torch.int16).numpy().tobytes()

    return packed, scale_bytes

# ── Main Conversion Function ──────────────────────────────────────────────────

def quantize_deepseek_dense(
    manifest_in: Path,
    dense_bin_in: Path,
    output_dir: Path,
    block_size: int = 32,
    quantize_embeddings: bool = True
):
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("  DeepSeek V4 Flash — INT4 Dense Quantizer (8GB GPU Mode)")
    print(f"  Source Manifest:  {manifest_in}")
    print(f"  Source Dense Bin: {dense_bin_in}")
    print(f"  Output Directory: {output_dir}")
    print(f"  Block Size:       {block_size}")
    print("=" * 70)

    with open(manifest_in, "r", encoding="utf-8") as f:
        base_manifest = json.load(f)

    dense_tensors = base_manifest.get("dense_tensors", {})
    if not dense_tensors:
        print("[ERROR] No dense_tensors found in manifest!")
        return

    # Map scale tensors to their corresponding weight tensor
    scale_to_weight = {}
    for name in dense_tensors:
        if name.endswith(".scale"):
            weight_name = name[:-6] + ".weight"
            if weight_name in dense_tensors:
                scale_to_weight[name] = weight_name

    out_bin_path = output_dir / "attention_dense_layers_q4.bin"
    out_manifest_path = output_dir / "moecher_manifest.json"

    dense_tensors_meta = {}
    dense_offset = 0
    t0 = time.time()

    total_orig_bytes = os.path.getsize(dense_bin_in)
    total_quant_bytes = 0

    with open(dense_bin_in, "rb") as f_in, open(out_bin_path, "wb") as f_out:
        processed_count = 0
        total_tensors = len(dense_tensors)

        for t_name, meta in dense_tensors.items():
            processed_count += 1

            # If this is a scale tensor belonging to an FP8 weight, it will be superseded by the INT4 scale
            if t_name in scale_to_weight:
                continue

            shape = meta.get("shape", [])
            dtype = meta.get("dtype", "BF16")
            offset = meta.get("offset", 0)
            nbytes = meta.get("nbytes", 0)

            f_in.seek(offset)
            raw_data = f_in.read(nbytes)

            is_2d_projection = (len(shape) == 2 and shape[0] >= 512 and shape[1] >= 512 and shape[1] % block_size == 0)
            is_fp8_weight = (dtype == "F8_E4M3")
            is_embed_or_head = ("embed" in t_name or "head" in t_name) and is_2d_projection

            should_quantize = (is_fp8_weight and is_2d_projection) or (quantize_embeddings and is_embed_or_head)

            if should_quantize:
                # 1. Recover float32 weight matrix
                if is_fp8_weight:
                    scale_name = t_name[:-7] + ".scale"
                    if scale_name in dense_tensors:
                        s_meta = dense_tensors[scale_name]
                        f_in.seek(s_meta["offset"])
                        scale_raw = f_in.read(s_meta["nbytes"])
                        scale_u8 = np.frombuffer(scale_raw, dtype=np.uint8)
                        weight_u8 = np.frombuffer(raw_data, dtype=np.uint8)
                        w_f32 = dequantize_fp8_e4m3_matrix(weight_u8, scale_u8, shape, block_size=128)
                    else:
                        weight_u8 = np.frombuffer(raw_data, dtype=np.uint8)
                        w_f32 = fp8_e4m3_to_float(weight_u8).reshape(shape)
                elif dtype == "BF16":
                    # Convert raw BF16 bytes to torch then numpy
                    w_t = torch.frombuffer(bytearray(raw_data), dtype=torch.bfloat16).reshape(shape)
                    w_f32 = w_t.to(torch.float32).numpy()
                else:
                    w_f32 = np.frombuffer(raw_data, dtype=np.float32).reshape(shape)

                # 2. Quantize to INT4 block-32
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

                if processed_count % 30 == 0 or processed_count == total_tensors:
                    print(f"[{processed_count}/{total_tensors}] INT4 Quantized: {t_name} {shape} -> {(w_nbytes + scale_nbytes) / (1024**2):.2f} MB")
            else:
                # Keep unquantized (norms, biases, compressors, sinks, HC tensors)
                f_out.write(raw_data)
                dense_tensors_meta[t_name] = {
                    "offset": dense_offset,
                    "nbytes": nbytes,
                    "dtype": dtype,
                    "shape": shape
                }
                dense_offset += len(raw_data)
                total_quant_bytes += len(raw_data)

    elapsed = time.time() - t0
    print("\n" + "=" * 70)
    print(f"  INT4 Dense Quantization Complete in {elapsed:.1f}s!")
    print(f"  Original Dense Bin:  {total_orig_bytes / (1024**3):.2f} GB")
    print(f"  Quantized Dense Bin: {total_quant_bytes / (1024**3):.2f} GB")
    print(f"  Dense VRAM Saved:    {(total_orig_bytes - total_quant_bytes) / (1024**3):.2f} GB (-{(1.0 - total_quant_bytes/total_orig_bytes)*100:.1f}%)")
    print("=" * 70)

    # Construct new manifest
    manifest = dict(base_manifest)
    manifest["dense_bin"] = "attention_dense_layers_q4.bin"
    manifest["dense_tensors"] = dense_tensors_meta

    # Ensure tokenizer.json is copied
    tok_src = manifest_in.parent / "tokenizer.json"
    if tok_src.exists():
        with open(output_dir / "tokenizer.json", "wb") as f_dst, open(tok_src, "rb") as f_src:
            f_dst.write(f_src.read())

    # Write manifest
    with open(out_manifest_path, "w", encoding="utf-8") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"\n[OK] Generated manifest: {out_manifest_path}")

    # Write Model Card README
    readme_content = f"""---
license: mit
tags:
- moecher
- quantized
- int4
- deepseek
- 8gb-gpu
pipeline_tag: text-generation
language:
- en
- zh
---

# DeepSeek V4 Flash INT4 Dense (8GB GPU Mode)

This repository contains the 4-bit quantized dense weights (`attention_dense_layers_q4.bin` ~3.9 GB) for **DeepSeek V4 Flash**, optimized to run in **8 GB VRAM** GPUs on the **Moecher Inference Engine**.

## Running with Moecher (8GB GPUs)

```bash
moecher.exe --manifest models/deepseek_v4_flash_q4/moecher_manifest.json --max-vram 6 --dram-cache-gb 64 --quiet
```
"""
    with open(output_dir / "README.md", "w", encoding="utf-8") as f_readme:
        f_readme.write(readme_content)

    print(f"[OK] Generated model card: {output_dir / 'README.md'}\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize DeepSeek V4 Flash Dense Layers to INT4 Block-32")
    parser.add_argument("--manifest-in", default="models/deepseek_v4_flash_iq2/moecher_manifest.json", help="Input manifest")
    parser.add_argument("--dense-bin-in", default="models/deepseek_v4_flash_iq2/attention_dense_layers.bin", help="Input dense bin")
    parser.add_argument("--output-dir", default="models/deepseek_v4_flash_q4", help="Output directory")
    parser.add_argument("--block-size", type=int, default=32, help="INT4 block size (default: 32)")
    parser.add_argument("--no-embed-quant", action="store_true", help="Do not quantize embeddings/head")

    args = parser.parse_args()

    quantize_deepseek_dense(
        Path(args.manifest_in),
        Path(args.dense_bin_in),
        Path(args.output_dir),
        block_size=args.block_size,
        quantize_embeddings=not args.no_embed_quant
    )
