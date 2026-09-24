#!/usr/bin/env python3
"""
quantize_blackwell.py — Pure Blackwell NVFP4 (FP4 E2M1) Quantizer for Qwen 3.8 / 2.5
Hardware Target: NVIDIA Blackwell architecture (Compute Capability >= 10.0 / 12.0)

Profiles:
  --target-vram 96gb  : Workstation profile (RTX PRO 6000 96GB). Full NVFP4 on transformer
                        projections, unquantized BF16 lm_head and embed_tokens (~19.5 GB total).
  --target-vram 32gb  : Enthusiast profile (RTX 5090 32GB). NVFP4 layers, BF16 lm_head (~17.0 GB total).
  --target-vram 16gb  : Mainstream profile (RTX 5060 Ti 16GB). Full NVFP4 including lm_head &
                        embed_tokens (~14.5 GB total, leaves ~1.5 GB headroom on 16GB cards).

All profiles include the 0.86 GB Vision Tower in unquantized BF16.
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

import torch
import numpy as np

# Midpoints between consecutive positive FP4 E2M1 values:
# [0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0]
# Midpoints: 0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0
FP4_MIDPOINTS_CPU = np.array([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], dtype=np.float32)


def quantize_matrix_nvfp4_block32_gpu(
    t: torch.Tensor,
    midpoints_gpu: torch.Tensor,
    block_size: int = 32
) -> Tuple[bytes, bytes]:
    """
    Quantizes a 2D weight matrix [rows, cols] to NVIDIA FP4 E2M1 with block-size 32 scaling.
    Scale factor is stored as E8M0 exponent (1 byte per block of 32 values).
    Returns:
        packed_bytes: packed uint8 bytes (low nibble = col 2k, high nibble = col 2k+1)
        scale_bytes: E8M0 exponent bytes [rows, cols // block_size]
    """
    rows, cols = t.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by {block_size}"
    num_blocks = cols // block_size

    b = t.view(rows, num_blocks, block_size).to(dtype=torch.float32)
    max_abs = torch.max(torch.abs(b), dim=-1, keepdim=True).values
    safe_max = torch.clamp(max_abs, min=1e-12)

    # Calculate E8M0 scale: 2^(E - 127) >= max_abs / 6.0
    exp = torch.clamp(torch.ceil(torch.log2(safe_max / 6.0)) + 127.0, 1.0, 254.0).to(torch.uint8)
    scale = torch.pow(2.0, exp.to(torch.float32) - 127.0)

    # Normalize values into [-6.0, 6.0]
    norm = b / scale
    sign = (norm < 0).to(torch.uint8)
    abs_norm = torch.abs(norm)

    # Map to closest FP4 magnitude index (0..7)
    idx = torch.bucketize(abs_norm, midpoints_gpu).to(torch.uint8)
    nibbles = (idx | (sign << 3)).view(rows, cols)

    # Pack 2 values per byte: low nibble = col 0, high nibble = col 1
    u_even = nibbles[:, 0::2]
    u_odd = nibbles[:, 1::2]
    packed = (u_even | (u_odd << 4)).contiguous()

    packed_bytes = packed.cpu().numpy().tobytes()
    scale_bytes = exp.squeeze(-1).contiguous().cpu().numpy().tobytes()
    return packed_bytes, scale_bytes


def quantize_matrix_nvfp4_block32_cpu(
    tensor_f32: np.ndarray,
    block_size: int = 32
) -> Tuple[bytes, bytes]:
    """CPU fallback implementation of NVFP4 quantization."""
    rows, cols = tensor_f32.shape
    assert cols % block_size == 0, f"cols {cols} must be divisible by {block_size}"
    num_blocks = cols // block_size

    b = tensor_f32.reshape(rows, num_blocks, block_size)
    max_abs = np.max(np.abs(b), axis=-1, keepdims=True)
    safe_max = np.maximum(max_abs, 1e-12)

    exp = np.clip(np.ceil(np.log2(safe_max / 6.0)) + 127, 1, 254).astype(np.uint8)
    scale = (2.0 ** (exp.astype(np.float32) - 127.0))

    norm = b / scale
    sign = (norm < 0).astype(np.uint8)
    abs_norm = np.abs(norm)

    idx = np.searchsorted(FP4_MIDPOINTS_CPU, abs_norm).astype(np.uint8)
    idx = np.clip(idx, 0, 7)
    nibbles = (idx | (sign << 3)).reshape(rows, cols)

    u_even = nibbles[:, 0::2]
    u_odd = nibbles[:, 1::2]
    packed = (u_even | (u_odd << 4)).tobytes()
    scale_bytes = exp.squeeze(-1).tobytes()
    return packed, scale_bytes


def read_safetensor_header(path: str):
    with open(path, "rb") as f:
        raw = f.read(8)
        header_size = struct.unpack("<Q", raw)[0]
        header_json = f.read(header_size)
    header = json.loads(header_json.decode("utf-8"))
    return header, 8 + header_size


def quantize_blackwell(
    input_dir: Path,
    output_dir: Path,
    target_vram: str = "96gb",
    block_size: int = 32,
    dry_run: bool = False
):
    output_dir.mkdir(parents=True, exist_ok=True)
    target_vram = target_vram.lower().strip()

    if target_vram not in ["96gb", "32gb", "16gb"]:
        raise ValueError(f"Unknown target-vram '{target_vram}'. Choose from: '96gb', '32gb', '16gb'.")

    # Locate source safetensors
    raw_dir = input_dir / "raw_hf"
    if not raw_dir.exists():
        raw_dir = input_dir

    st_files = sorted(list(raw_dir.glob("*.safetensors")))
    if not st_files:
        raise FileNotFoundError(f"No safetensors found in {raw_dir}")

    # CUDA device setup
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")
    midpoints_gpu = torch.tensor([0.25, 0.75, 1.25, 1.75, 2.5, 3.5, 5.0], device=device, dtype=torch.float32) if use_cuda else None

    print("═" * 78)
    print("  MinnieTheMoECher — Blackwell Native NVFP4 (FP4 E2M1) Quantizer")
    print(f"  Target VRAM Profile : {target_vram.upper()}")
    print(f"  Hardware Device     : {torch.cuda.get_device_name(0) if use_cuda else 'CPU'}")
    print(f"  Input Source        : {raw_dir}")
    print(f"  Output Directory    : {output_dir}")
    print("═" * 78)

    # Determine profile policies
    if target_vram == "96gb":
        quantize_embed = False
        quantize_head = False
        quantize_mtp = False
        profile_desc = "Full NVFP4 transformer layers + Unquantized BF16 lm_head, embed & MTP (~19.5 GB)"
    elif target_vram == "32gb":
        quantize_embed = True
        quantize_head = False
        quantize_mtp = True
        profile_desc = "Full NVFP4 transformer layers + Unquantized BF16 lm_head, NVFP4 embed (~17.0 GB)"
    else: # 16gb
        quantize_embed = True
        quantize_head = True
        quantize_mtp = True
        profile_desc = "Pure NVFP4 (W4A4) on ALL layers including lm_head & embed (~14.5 GB)"

    print(f"\nConfiguration: {profile_desc}\n")

    # Index all tensors across safetensors
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

    total_tensors = len(tensor_index)
    print(f"Indexed {total_tensors} tensors across {len(st_files)} safetensor files.\n")

    out_bin_path = output_dir / "attention_dense_layers_nvfp4.bin"
    out_manifest_path = output_dir / "moecher_manifest.json"

    dense_tensors_meta = {}
    dense_offset = 0
    t0 = time.time()
    total_orig_bytes = sum(m["nbytes"] for m in tensor_index.values())
    total_quant_bytes = 0

    if dry_run:
        print("[Dry Run] Simulating quantization without writing binary...")

    out_file = open(out_bin_path, "wb") if not dry_run else None

    try:
        for idx, (t_name, meta) in enumerate(tensor_index.items()):
            shape = meta["shape"]
            dtype = meta["dtype"]
            nbytes = meta["nbytes"]

            with open(meta["file"], "rb") as f_in:
                f_in.seek(meta["data_offset"])
                raw_data = f_in.read(nbytes)

            is_2d = (len(shape) == 2 and shape[0] >= 512 and shape[1] >= 512 and shape[1] % block_size == 0)
            is_visual = ("visual" in t_name)
            is_embed = ("embed_tokens" in t_name and "mtp" not in t_name)
            is_head = ("lm_head" in t_name or "head.weight" in t_name)
            is_mtp = ("mtp" in t_name)
            is_norm = ("norm" in t_name or "bias" in t_name or "conv1d" in t_name or "dt_bias" in t_name or "A_log" in t_name)
            is_proj_ab = ("in_proj_a" in t_name or "in_proj_b" in t_name)

            should_quantize = False
            if is_visual or is_norm or is_proj_ab or not is_2d:
                should_quantize = False
            elif is_embed:
                should_quantize = quantize_embed
            elif is_head:
                should_quantize = quantize_head
            elif is_mtp:
                should_quantize = quantize_mtp
            else:
                # All standard attention & MLP projections
                should_quantize = True

            if should_quantize:
                # Quantize with NVFP4 E2M1
                if use_cuda:
                    t_bf16 = torch.frombuffer(bytearray(raw_data), dtype=torch.bfloat16).reshape(shape).to(device)
                    packed_w, packed_s = quantize_matrix_nvfp4_block32_gpu(t_bf16, midpoints_gpu, block_size)
                else:
                    t_f32 = np.frombuffer(raw_data, dtype=np.uint16)
                    # Convert BF16 uint16 to float32
                    u32 = t_f32.astype(np.uint32) << 16
                    f32 = u32.view(np.float32).reshape(shape)
                    packed_w, packed_s = quantize_matrix_nvfp4_block32_cpu(f32, block_size)

                w_offset = dense_offset
                w_nbytes = len(packed_w)
                if out_file:
                    out_file.write(packed_w)
                dense_offset += w_nbytes

                scale_offset = dense_offset
                scale_nbytes = len(packed_s)
                if out_file:
                    out_file.write(packed_s)
                dense_offset += scale_nbytes

                total_quant_bytes += (w_nbytes + scale_nbytes)

                dense_tensors_meta[t_name] = {
                    "offset": w_offset,
                    "nbytes": w_nbytes,
                    "dtype": "fp4",
                    "shape": shape,
                    "scale_offset": scale_offset,
                    "scale_nbytes": scale_nbytes,
                    "scale_dtype": "F8_E8M0",
                    "block_size": block_size
                }

                if idx % 50 == 0 or idx == total_tensors - 1:
                    cur_mb = (w_nbytes + scale_nbytes) / (1024**2)
                    orig_mb = nbytes / (1024**2)
                    print(f"[{idx+1:4d}/{total_tensors}] NVFP4 : {t_name:<55} {str(shape):<16} {orig_mb:6.1f}MB -> {cur_mb:6.1f}MB")
            else:
                # Store unquantized (BF16)
                if out_file:
                    out_file.write(raw_data)
                dense_tensors_meta[t_name] = {
                    "offset": dense_offset,
                    "nbytes": nbytes,
                    "dtype": dtype,
                    "shape": shape
                }
                dense_offset += nbytes
                total_quant_bytes += nbytes

                if (idx % 50 == 0 or idx == total_tensors - 1) and (is_head or is_embed or is_visual):
                    print(f"[{idx+1:4d}/{total_tensors}] BF16  : {t_name:<55} {str(shape):<16} {nbytes/(1024**2):6.1f}MB (unquantized)")

    finally:
        if out_file:
            out_file.close()

    elapsed = time.time() - t0
    print("\n" + "═" * 78)
    print(f"  Quantization Complete in {elapsed:.1f}s ({elapsed/60.0:.2f} min)!")
    print(f"  Original BF16 Model Size : {total_orig_bytes / (1024**3):.2f} GB")
    print(f"  Quantized NVFP4 Size     : {total_quant_bytes / (1024**3):.2f} GB")
    print(f"  Effective Compression    : {total_orig_bytes / total_quant_bytes:.2f}x")
    print("═" * 78)

    # Read base manifest if available to copy model_config
    base_manifest_path = input_dir / "moecher_manifest_qwen.json"
    if not base_manifest_path.exists():
        base_manifest_path = input_dir.parent / "qwen3_8_27b" / "moecher_manifest_qwen.json"

    if base_manifest_path.exists():
        with open(base_manifest_path, "r") as f:
            base_manifest = json.load(f)
        model_config = base_manifest.get("model_config", {})
    else:
        # Generate default Qwen 3.8 27B config
        model_config = {
            "architecture": "qwen2",
            "vocab_size": 248320,
            "hidden_size": 5120,
            "num_hidden_layers": 64,
            "num_attention_heads": 24,
            "num_key_value_heads": 4,
            "head_dim": 256,
            "intermediate_size": 17408,
            "rms_norm_eps": 1e-06,
            "rope_theta": 10000000.0,
            "max_seq_len": 32768,
            "eos_token_id": 248046,
            "expert_dtype": "none",
            "has_vision": True
        }

    model_config["has_vision"] = True

    manifest = {
        "model_config": model_config,
        "visual_config": {
            "depth": 32,
            "embed_dim": 1152,
            "num_heads": 16,
            "patch_size": 14,
            "spatial_merge_size": 2
        },
        "tokenizer": {"tokenizer_json": "tokenizer.json"},
        "dense_bin": "attention_dense_layers_nvfp4.bin",
        "expert_bin": "",
        "target_vram_profile": target_vram,
        "dense_tensors": dense_tensors_meta,
        "expert_layout": {
            "block_size": 0,
            "n_layers": 0,
            "n_experts": 0,
            "parts": {}
        }
    }

    if not dry_run:
        with open(out_manifest_path, "w") as f:
            json.dump(manifest, f, indent=2)
        print(f"Wrote manifest: {out_manifest_path}")

        # Copy tokenizer & configs
        for fname in ["tokenizer.json", "config.json", "preprocessor_config.json"]:
            for src_cand in [raw_dir / fname, input_dir / fname, input_dir.parent / "qwen3_8_27b" / fname]:
                if src_cand.exists():
                    shutil.copy2(src_cand, output_dir / fname)
                    print(f"Copied {fname} -> {output_dir / fname}")
                    break

    print(f"\nModel ready at: {output_dir}")
    print(f"Manifest file : {out_manifest_path}\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MinnieTheMoECher — Blackwell NVFP4 Multi-Profile Quantizer")
    parser.add_argument("--input-dir", default="models/qwen3_8_27b", help="Source directory containing raw_hf or safetensors")
    parser.add_argument("--output-dir", required=True, help="Destination directory for NVFP4 model")
    parser.add_argument("--target-vram", choices=["96gb", "32gb", "16gb"], default="96gb",
                        help="Target VRAM optimization profile (96gb, 32gb, 16gb)")
    parser.add_argument("--block-size", type=int, default=32, help="Block size for scale factor (default: 32)")
    parser.add_argument("--dry-run", action="store_true", help="Simulate without writing output files")

    args = parser.parse_args()
    quantize_blackwell(
        input_dir=Path(args.input_dir),
        output_dir=Path(args.output_dir),
        target_vram=args.target_vram,
        block_size=args.block_size,
        dry_run=args.dry_run
    )
