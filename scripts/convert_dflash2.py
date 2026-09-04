#!/usr/bin/env python3
"""
Convert incoai/Qwen3.8-27B-DFlash2 to MinnieTheMoEcher binary & manifest format.
Supports INT4 symmetric block quantization (block size = 32) and BF16.
"""

import os
import sys
import json
import argparse
import numpy as np
import ml_dtypes
from safetensors import safe_open
from huggingface_hub import hf_hub_download

def quantize_int4_symmetric(tensor_f32, block_size=32):
    """
    Quantizes a 2D float32 weight matrix [N, K] to INT4 symmetric with zero-point 8.
    Returns (packed_uint8 [N, K/2], scale_bf16 [N, K/32]).
    """
    N, K = tensor_f32.shape
    assert K % block_size == 0, f"K ({K}) must be divisible by block_size ({block_size})"
    num_blocks = K // block_size

    reshaped = tensor_f32.reshape(N, num_blocks, block_size)
    max_val = np.max(np.abs(reshaped), axis=-1, keepdims=True)
    scale = np.maximum(max_val / 7.0, 1e-8)  # INT4 signed range [-7, 7] mapped to [1, 15], 0 is reserved/clamp

    # Quantize to [0, 15] centered at 8
    quantized = np.clip(np.round(reshaped / scale) + 8.0, 0, 15).astype(np.uint8)
    quantized = quantized.reshape(N, K)

    # Pack pairs of 4-bit nibbles into 1 byte (low nibble in bits 0..3, high nibble in bits 4..7)
    low_nibble = quantized[:, 0::2] & 0x0F
    high_nibble = quantized[:, 1::2] & 0x0F
    packed_weights = (low_nibble | (high_nibble << 4)).astype(np.uint8)

    scale_bf16 = scale.reshape(N, num_blocks).astype(ml_dtypes.bfloat16)
    return packed_weights, scale_bf16

def main():
    parser = argparse.ArgumentParser(description="Convert DFlash2 to MinnieTheMoEcher format")
    parser.add_argument("--repo-id", default="incoai/Qwen3.8-27B-DFlash2", help="HuggingFace repo ID")
    parser.add_argument("--output-dir", default="f:/Moecher/models/qwen3_8_27b_q4/dflash2", help="Output directory")
    parser.add_argument("--quant", default="int4", choices=["int4", "bf16"], help="Weight format (int4 or bf16)")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Downloading DFlash2 model from {args.repo_id} ===")
    config_path = hf_hub_download(args.repo_id, "config.json")
    model_path = hf_hub_download(args.repo_id, "model.safetensors")

    config = json.load(open(config_path, "r", encoding="utf-8"))
    print(f"Loaded config: {config.get('num_hidden_layers', 5)} layers, hidden={config.get('hidden_size', 5120)}")

    st = safe_open(model_path, framework="numpy")
    bin_path = os.path.join(args.output_dir, "dflash2_layers.bin")
    manifest_path = os.path.join(args.output_dir, "dflash2_manifest.json")

    manifest = {
        "model_config": config,
        "dense_bin": os.path.basename(bin_path),
        "dense_tensors": {}
    }

    print(f"=== Converting tensors to {bin_path} (quant={args.quant}) ===")
    current_offset = 0

    with open(bin_path, "wb") as f_out:
        for key in sorted(list(st.keys())):
            tensor = st.get_tensor(key)
            print(f"Processing {key}: shape {tensor.shape}, dtype {tensor.dtype}")

            if args.quant == "int4" and len(tensor.shape) == 2 and tensor.shape[-1] % 32 == 0 and "norm" not in key and "bias" not in key and "codebook" not in key and "conv" not in key:
                # Quantize 2D projection weights to INT4
                tensor_f32 = tensor.astype(np.float32)
                packed_w, scale_bf16 = quantize_int4_symmetric(tensor_f32, block_size=32)

                # Write packed INT4 weights
                w_bytes = packed_w.tobytes()
                w_offset = current_offset
                f_out.write(w_bytes)
                current_offset += len(w_bytes)

                # Write scales
                s_bytes = scale_bf16.tobytes()
                s_offset = current_offset
                f_out.write(s_bytes)
                current_offset += len(s_bytes)

                manifest["dense_tensors"][key] = {
                    "offset": w_offset,
                    "nbytes": len(w_bytes),
                    "dtype": "int4",
                    "shape": list(tensor.shape),
                    "scale_offset": s_offset,
                    "scale_nbytes": len(s_bytes)
                }
            else:
                # Write as BF16
                tensor_bf16 = tensor.astype(ml_dtypes.bfloat16)
                raw_bytes = tensor_bf16.tobytes()
                tensor_offset = current_offset
                f_out.write(raw_bytes)
                current_offset += len(raw_bytes)

                manifest["dense_tensors"][key] = {
                    "offset": tensor_offset,
                    "nbytes": len(raw_bytes),
                    "dtype": "BF16",
                    "shape": list(tensor.shape)
                }

    with open(manifest_path, "w", encoding="utf-8") as f_m:
        json.dump(manifest, f_m, indent=2)

    total_mb = current_offset / (1024 * 1024)
    print(f"\n[SUCCESS] DFlash2 converted successfully!")
    print(f"Binary:   {bin_path} ({total_mb:.2f} MB)")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
