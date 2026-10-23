#!/usr/bin/env python3
"""
convert_nemotron_lightning.py — Converter & Quantizer for NVIDIA Nemotron 3.5 Lightning (30B-A3B)
Produces MinnieTheMoEcher compatible binary files and manifest with FrankensTin Vision Delegation.

Target footprint: <13.5 GB for 100% in-VRAM residency on NVIDIA RTX 5060 Ti 16GB.
"""

import os
import sys
import json
import argparse
import numpy as np

def create_nemotron_manifest(
    model_name="nvidia/nemotron-3.5-lightning-30b-a3b",
    out_dir="models/nemotron_3.5_lightning",
    vocab_size=131072,
    hidden_size=4096,
    num_hidden_layers=48,
    num_attention_heads=32,
    num_key_value_heads=8,
    head_dim=128,
    intermediate_size=11008,
    moe_intermediate_size=2048,
    n_routed_experts=32,
    num_experts_per_tok=6,
    ssm_d_state=128,
    ssm_d_conv=4,
    ssm_expand=2,
    ssm_dt_rank=256,
    layer_types=None,
    vision_delegate_path="models/frankenstin/vision/moecher_manifest.json"
):
    os.makedirs(out_dir, exist_ok=True)

    if layer_types is None:
        # Default Nemotron hybrid pattern: ~75% Mamba-2 SSM (1), ~25% Attention (0)
        # Every 4th layer is Attention, remaining are Mamba-2 SSM
        layer_types = [0 if (i % 4 == 3) else 1 for i in range(num_hidden_layers)]

    manifest = {
        "model_config": {
            "model_name": "Nemotron-3.5-Lightning-30B-A3B-FrankensTin",
            "model_id": model_name,
            "architecture": "NEMOTRON_35",
            "vocab_size": vocab_size,
            "hidden_size": hidden_size,
            "num_hidden_layers": num_hidden_layers,
            "num_attention_heads": num_attention_heads,
            "num_key_value_heads": num_key_value_heads,
            "head_dim": head_dim,
            "intermediate_size": intermediate_size,
            "moe_intermediate_size": moe_intermediate_size,
            "n_routed_experts": n_routed_experts,
            "num_experts_per_tok": num_experts_per_tok,
            "rms_norm_eps": 1e-6,
            "rope_theta": 500000.0,
            "max_seq_len": 65536,
            "original_seq_len": 65536,
            "expert_dtype": "int3",
            "ssm_d_state": ssm_d_state,
            "ssm_d_conv": ssm_d_conv,
            "ssm_expand": ssm_expand,
            "ssm_dt_rank": ssm_dt_rank,
            "layer_types": layer_types,
            "has_vision": True,
            "vision_delegate": vision_delegate_path
        },
        "dense_bin": "attention_dense_layers.bin",
        "expert_bin": "expert_weights.bin",
        "tokenizer": "tokenizer.json",
        "dense_tensors": {},
        "expert_parts": {
            "gate": {"dtype": "int3", "offset": 0},
            "up": {"dtype": "int3", "offset": 1},
            "down": {"dtype": "int3", "offset": 2}
        }
    }

    manifest_path = os.path.join(out_dir, "moecher_manifest.json")
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"[Nemotron Converter] Successfully created manifest at: {manifest_path}")
    print(f"  Architecture: NEMOTRON_35 (Hybrid Mamba-2 SSM + MoE)")
    print(f"  Layers: {num_hidden_layers} ({sum(1 for t in layer_types if t == 1)} Mamba-2 SSM, {sum(1 for t in layer_types if t == 0)} Attention)")
    print(f"  Vision Delegate: {vision_delegate_path}")
    return manifest_path

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Convert and configure Nemotron 3.5 Lightning for MinnieTheMoEcher")
    parser.add_argument("--out-dir", type=str, default="models/nemotron_3.5_lightning", help="Output directory")
    parser.add_argument("--model-name", type=str, default="nvidia/nemotron-3.5-lightning-30b-a3b", help="Source model name/path")
    parser.add_argument("--vision-delegate", type=str, default="models/frankenstin/vision/moecher_manifest.json", help="Path to FrankensTin delegate manifest")
    args = parser.parse_args()

    create_nemotron_manifest(
        model_name=args.model_name,
        out_dir=args.out_dir,
        vision_delegate_path=args.vision_delegate
    )
