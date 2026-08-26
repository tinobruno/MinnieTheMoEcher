#!/usr/bin/env python3
"""
quantize_qwen.py — Quantize Qwen 3.8 / 2.5 weights to 4-bit (INT4 block-32)
for MinnieTheMoECher to run in ~14-15 GB VRAM at 80-100+ tok/s.

Usage:
    python3 scripts/quantize_qwen.py --input-dir models/qwen3_8_27b \
                                    --output-dir models/qwen3_8_27b_q4 \
                                    [--block-size 32]
"""

import os
import sys
import json
import time
import struct
import argparse
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

import torch
import numpy as np

def quantize_matrix_int4_block32(tensor: torch.Tensor, block_size: int = 32):
    """
    Quantizes a 2D weight matrix [rows, cols] to symmetric INT4 with block size 32.
    Returns:
        packed_bytes: numpy.ndarray of uint8, shape [rows, cols // 2]
        scale_bytes: numpy.ndarray of bfloat16, shape [rows, cols // block_size]
    """
    rows, cols = tensor.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by {block_size}"
    num_blocks = cols // block_size

    # Reshape into [rows, num_blocks, block_size]
    t = tensor.to(device="cuda" if torch.cuda.is_available() else "cpu", dtype=torch.float32)
    t_blocks = t.view(rows, num_blocks, block_size)

    # Scale per block: max_abs / 7.0
    max_abs = torch.max(torch.abs(t_blocks), dim=-1, keepdim=True).values
    scale = torch.clamp(max_abs / 7.0, min=1e-8) # [rows, num_blocks, 1]

    # Quantize to [-8, 7]
    q = torch.clamp(torch.round(t_blocks / scale), -8, 7).to(torch.int32)
    
    # Offset by +8 to get [0, 15]
    u = (q + 8).to(torch.uint8).view(rows, cols)

    # Pack 2 values per byte: low nibble = col 2k, high nibble = col 2k+1
    u_even = u[:, 0::2]
    u_odd = u[:, 1::2]
    packed = u_even | (u_odd << 4)

    # Scale in BF16 bytes
    scale_bf16 = scale.squeeze(-1).to(torch.bfloat16)
    packed_np = packed.cpu().numpy()
    scale_bytes = scale_bf16.view(torch.int16).cpu().numpy().tobytes()

    return packed_np.tobytes(), scale_bytes

def should_quantize_tensor(name: str, shape: list, block_size: int = 32) -> bool:
    # Only quantize 2D matrices in LLM transformer layers
    if len(shape) != 2:
        return False
    if not ("model.layers." in name or "model.language_model.layers." in name or "layers." in name):
        return False
    if "embed_tokens" in name or "lm_head" in name:
        return False
    if "norm" in name or "conv1d" in name:
        return False
    if "in_proj_a" in name or "in_proj_b" in name:
        return False
    if shape[1] % block_size != 0:
        return False
    # Only quantize projection matrices
    if shape[0] >= 512 and shape[1] >= 512:
        return True
    return False

def read_safetensor_header(path: str):
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json.decode("utf-8"))
    return header, 8 + header_size

def quantize_qwen(input_dir: Path, output_dir: Path, block_size: int = 32):
    output_dir.mkdir(parents=True, exist_ok=True)

    # Check for raw_hf safetensors
    raw_dir = input_dir / "raw_hf"
    if not raw_dir.exists():
        raw_dir = input_dir

    st_files = sorted(list(raw_dir.glob("*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors found in {raw_dir}")

    # Read base manifest / config
    base_manifest_path = input_dir / "moecher_manifest_qwen.json"
    if base_manifest_path.exists():
        with open(base_manifest_path, "r") as f:
            base_manifest = json.load(f)
    else:
        raise FileNotFoundError(f"Base manifest not found: {base_manifest_path}")

    print(f"═══ Quantizing Qwen 3.8 to INT4 (Block Size = {block_size}) ═══")
    print(f"Source: {raw_dir}")
    print(f"Target: {output_dir}\n")

    # Index all safetensors
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

    out_bin_path = output_dir / "attention_dense_layers_q4.bin"
    out_manifest_path = output_dir / "moecher_manifest_qwen_q4.json"

    dense_tensors_meta = {}
    dense_offset = 0
    t0 = time.time()

    total_orig_bytes = 0
    total_quant_bytes = 0

    with open(out_bin_path, "wb") as f_out:
        for idx, (t_name, meta) in enumerate(tensor_index.items()):
            shape = meta["shape"]
            dtype = meta["dtype"]
            total_orig_bytes += meta["nbytes"]

            # Read tensor
            with open(meta["file"], "rb") as f_in:
                f_in.seek(meta["data_offset"])
                raw_data = f_in.read(meta["nbytes"])

            if should_quantize_tensor(t_name, shape, block_size):
                # Load as PyTorch tensor
                # convert bfloat16 raw bytes to tensor
                t = torch.frombuffer(bytearray(raw_data), dtype=torch.bfloat16).reshape(shape)
                packed_w, packed_scale = quantize_matrix_int4_block32(t, block_size)

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

                if idx % 30 == 0 or idx == len(tensor_index) - 1:
                    print(f"[{idx+1}/{len(tensor_index)}] Quantized {t_name} {shape}: {meta['nbytes']/(1024**2):.1f}MB -> {(w_nbytes+scale_nbytes)/(1024**2):.1f}MB")
            else:
                # Store unquantized (e.g. embed_tokens, norms)
                f_out.write(raw_data)
                dense_tensors_meta[t_name] = {
                    "offset": dense_offset,
                    "nbytes": meta["nbytes"],
                    "dtype": dtype,
                    "shape": shape
                }
                dense_offset += len(raw_data)
                total_quant_bytes += len(raw_data)

    elapsed = time.time() - t0
    print(f"\nQuantization Complete in {elapsed:.1f}s!")
    print(f"Original Model Size: {total_orig_bytes / (1024**3):.2f} GB")
    print(f"Quantized Model Size: {total_quant_bytes / (1024**3):.2f} GB")
    print(f"Compression Ratio: {total_orig_bytes / total_quant_bytes:.2f}x\n")

    # Build manifest
    manifest = dict(base_manifest)
    manifest["dense_bin"] = "attention_dense_layers_q4.bin"
    manifest["dense_tensors"] = dense_tensors_meta

    # Copy tokenizer.json
    tok_src = input_dir / "tokenizer.json"
    if tok_src.exists():
        with open(output_dir / "tokenizer.json", "wb") as f_dst, open(tok_src, "rb") as f_src:
            f_dst.write(f_src.read())

    with open(out_manifest_path, "w") as f_man:
        json.dump(manifest, f_man, indent=2)

    print(f"Wrote manifest to: {out_manifest_path}")
    print(f"Ready to run with:")
    print(f"  ./build/moecher --manifest {out_manifest_path} --port 8000 --quiet\n")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Quantize Qwen 3.8 to INT4 Block-32")
    parser.add_argument("--input-dir", default="models/qwen3_8_27b", help="Source Qwen directory")
    parser.add_argument("--output-dir", default="models/qwen3_8_27b_q4", help="Target output directory")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for scale factor")
    args = parser.parse_args()

    quantize_qwen(Path(args.input_dir), Path(args.output_dir), args.block_size)
