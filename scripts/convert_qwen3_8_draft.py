#!/usr/bin/env python3
"""
Convert empero-ai/Qwen3.8-2B-Distill to MinnieTheMoEcher INT4 draft format.
Produces draft_manifest.json and qwen3_8_draft_2b.bin
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
    scale = np.maximum(max_val / 7.0, 1e-8)  # INT4 signed range [-7, 7] mapped to [1, 15]

    quantized = np.clip(np.round(reshaped / scale) + 8.0, 0, 15).astype(np.uint8)
    quantized = quantized.reshape(N, K)

    low_nibble = quantized[:, 0::2] & 0x0F
    high_nibble = quantized[:, 1::2] & 0x0F
    packed_weights = (low_nibble | (high_nibble << 4)).astype(np.uint8)

    scale_bf16 = scale.reshape(N, num_blocks).astype(ml_dtypes.bfloat16)
    return packed_weights, scale_bf16

def main():
    parser = argparse.ArgumentParser(description="Convert Qwen3.8-2B-Distill to MinnieTheMoEcher format")
    parser.add_argument("--repo-id", default="empero-ai/Qwen3.8-2B-Distill", help="HuggingFace repo ID")
    parser.add_argument("--output-dir", default="f:/Moecher/models/qwen3_8_27b_q4/draft_2b", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    print(f"=== Downloading config for {args.repo_id} ===")
    config_path = hf_hub_download(args.repo_id, "config.json")
    model_path = hf_hub_download(args.repo_id, "model.safetensors")

    config = json.load(open(config_path, "r", encoding="utf-8"))
    text_cfg = config.get("text_config", config)
    
    bin_path = os.path.join(args.output_dir, "qwen3_8_draft_2b.bin")
    manifest_path = os.path.join(args.output_dir, "draft_manifest.json")

    manifest = {
        "model_config": {
            "architecture": "qwen3_8_draft",
            "vocab_size": text_cfg.get("vocab_size", 248320),
            "hidden_size": text_cfg.get("hidden_size", 2048),
            "num_hidden_layers": text_cfg.get("num_hidden_layers", 24),
            "num_attention_heads": text_cfg.get("num_attention_heads", 8),
            "num_key_value_heads": text_cfg.get("num_key_value_heads", 2),
            "head_dim": text_cfg.get("head_dim", 256),
            "intermediate_size": text_cfg.get("intermediate_size", 6144),
            "linear_num_heads": text_cfg.get("linear_num_key_heads", 16),
            "linear_head_dim": text_cfg.get("linear_key_head_dim", 128),
            "rms_norm_eps": text_cfg.get("rms_norm_eps", 1e-6),
            "rope_theta": 10000000.0,
            "layer_types": text_cfg.get("layer_types", [])
        },
        "dense_bin": os.path.basename(bin_path),
        "dense_tensors": {}
    }

    print(f"=== Converting tensors to {bin_path} (INT4 quant) ===")
    st = safe_open(model_path, framework="numpy")
    current_offset = 0

    with open(bin_path, "wb") as f_out:
        # Only process language_model tensors
        for key in sorted(list(st.keys())):
            if not key.startswith("model.language_model.") and not key.startswith("lm_head."):
                continue

            tensor = st.get_tensor(key)
            print(f"Processing {key}: shape {tensor.shape}, dtype {tensor.dtype}")

            is_2d_proj = (len(tensor.shape) == 2 and 
                          tensor.shape[-1] % 32 == 0 and 
                          "norm" not in key and 
                          "bias" not in key and 
                          "conv" not in key and 
                          "embed" not in key)

            if is_2d_proj:
                # Quantize 2D projection weights to INT4
                tensor_f32 = tensor.astype(np.float32)
                packed_w, scale_bf16 = quantize_int4_symmetric(tensor_f32, block_size=32)

                w_bytes = packed_w.tobytes()
                w_offset = current_offset
                f_out.write(w_bytes)
                current_offset += len(w_bytes)

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
    print(f"\n[SUCCESS] Qwen 3.8 Draft Model converted successfully!")
    print(f"Binary:   {bin_path} ({total_mb:.2f} MB)")
    print(f"Manifest: {manifest_path}")

if __name__ == "__main__":
    main()
