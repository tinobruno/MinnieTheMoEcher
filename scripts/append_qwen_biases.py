#!/usr/bin/env python3
import json
import os
from pathlib import Path
import torch
from safetensors import safe_open

def main():
    snap_dir = Path("/home/tinobruno/.cache/huggingface/hub/models--Qwen--Qwen2.5-VL-3B-Instruct/snapshots/66285546d2b821cf421d4f5eb2576359d3770cd3")
    out_dir = Path("models/frankenstin/vision")
    dense_bin_path = out_dir / "attention_dense_layers.bin"
    manifest_path = out_dir / "moecher_manifest.json"

    print(f"Loading manifest from {manifest_path}...")
    with open(manifest_path, "r") as f:
        manifest = json.load(f)

    dense_meta = manifest["dense_tensors"]

    # Index safetensors
    st_files = sorted(list(snap_dir.glob("*.safetensors")))
    tensor_map = {}
    for st_path in st_files:
        with safe_open(str(st_path), framework="pt") as f:
            for k in f.keys():
                tensor_map[k] = str(st_path)

    def get_tensor(name: str) -> torch.Tensor:
        st_path = tensor_map[name]
        with safe_open(st_path, framework="pt") as f:
            return f.get_tensor(name)

    # Open binary file in append mode
    file_size = os.path.getsize(dense_bin_path)
    print(f"Current dense bin size: {file_size} bytes ({file_size / (1024*1024):.2f} MB)")

    curr_offset = file_size
    biases_appended = 0

    with open(dense_bin_path, "ab") as fout:
        for layer_idx in range(36):
            pfx = f"model.layers.{layer_idx}."
            for proj in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"]:
                bias_name = pfx + proj + ".bias"
                if bias_name in dense_meta:
                    print(f"Skipping already present tensor: {bias_name}")
                    continue

                if bias_name not in tensor_map:
                    print(f"Warning: {bias_name} not found in safetensors!")
                    continue

                t = get_tensor(bias_name).to(torch.bfloat16)
                data = t.view(torch.int16).cpu().numpy().tobytes()

                # Align to 64 bytes
                align = (64 - (curr_offset % 64)) % 64
                if align > 0:
                    fout.write(b"\x00" * align)
                    curr_offset += align

                offset = curr_offset
                fout.write(data)
                nbytes = len(data)
                curr_offset += nbytes

                dense_meta[bias_name] = {
                    "offset": offset,
                    "nbytes": nbytes,
                    "dtype": "BF16",
                    "shape": list(t.shape)
                }
                biases_appended += 1

    print(f"Appended {biases_appended} bias tensors. New file size: {curr_offset} bytes ({curr_offset / (1024*1024):.2f} MB)")

    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print("Updated moecher_manifest.json successfully!")

if __name__ == "__main__":
    main()
