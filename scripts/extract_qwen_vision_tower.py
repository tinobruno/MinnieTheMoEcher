#!/usr/bin/env python3
import json
import os
import sys

def main():
    qwen_manifest_path = "models/qwen3_8_27b/moecher_manifest_qwen.json"
    qwen_dense_bin = "models/qwen3_8_27b/attention_dense_layers.bin"
    output_bin = "models/frankenstin/qwen_vision_tower.bin"
    output_json = "models/frankenstin/qwen_vision_tensors.json"

    print(f"Loading Qwen manifest from {qwen_manifest_path}...")
    with open(qwen_manifest_path, "r") as f:
        qwen_manifest = json.load(f)

    dense_tensors = qwen_manifest["dense_tensors"]

    # We extract all model.visual.* tensors EXCEPT linear_fc2 (which is replaced by our 4096-dim bridge)
    vit_tensor_names = []
    # 1. Patch & Pos embed
    vit_tensor_names.extend([
        "model.visual.patch_embed.proj.weight",
        "model.visual.patch_embed.proj.bias",
        "model.visual.pos_embed.weight"
    ])
    # 2. 27 Blocks
    for i in range(27):
        prefix = f"model.visual.blocks.{i}."
        vit_tensor_names.extend([
            prefix + "norm1.weight",
            prefix + "norm1.bias",
            prefix + "attn.qkv.weight",
            prefix + "attn.qkv.bias",
            prefix + "attn.proj.weight",
            prefix + "attn.proj.bias",
            prefix + "norm2.weight",
            prefix + "norm2.bias",
            prefix + "mlp.linear_fc1.weight",
            prefix + "mlp.linear_fc1.bias",
            prefix + "mlp.linear_fc2.weight",
            prefix + "mlp.linear_fc2.bias",
        ])
    # 3. Spatial merger up to fc1
    vit_tensor_names.extend([
        "model.visual.merger.norm.weight",
        "model.visual.merger.norm.bias",
        "model.visual.merger.linear_fc1.weight",
        "model.visual.merger.linear_fc1.bias"
    ])

    print(f"Total ViT tensors to extract: {len(vit_tensor_names)}")

    new_tensor_map = {}
    current_offset = 0

    os.makedirs(os.path.dirname(output_bin), exist_ok=True)

    with open(qwen_dense_bin, "rb") as fin, open(output_bin, "wb") as fout:
        for name in vit_tensor_names:
            if name not in dense_tensors:
                print(f"ERROR: Tensor {name} missing from Qwen manifest!")
                sys.exit(1)
            meta = dense_tensors[name]
            offset = meta["offset"]
            nbytes = meta["nbytes"]

            fin.seek(offset)
            data = fin.read(nbytes)
            if len(data) != nbytes:
                print(f"ERROR: Failed to read {nbytes} bytes for {name} at {offset}!")
                sys.exit(1)

            fout.write(data)
            new_tensor_map[name] = {
                "offset": current_offset,
                "nbytes": nbytes,
                "dtype": meta.get("dtype", "BF16"),
                "shape": meta.get("shape", [])
            }
            current_offset += nbytes

    print(f"Successfully extracted {len(new_tensor_map)} tensors to {output_bin} ({current_offset / (1024*1024):.2f} MB)")
    with open(output_json, "w") as f:
        json.dump(new_tensor_map, f, indent=2)
    print(f"Saved tensor metadata to {output_json}")

if __name__ == "__main__":
    main()
