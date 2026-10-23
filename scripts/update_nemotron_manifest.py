#!/usr/bin/env python3
import json
import struct
from pathlib import Path

def update_manifest():
    manifest_p = Path("models/nemotron_3.5_lightning/moecher_manifest.json")
    dense_bin_p = Path("models/nemotron_3.5_lightning/attention_dense_layers.bin")

    if not manifest_p.exists() or not dense_bin_p.exists():
        print("Error: manifest or dense_bin not found!")
        return

    with open(manifest_p, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    dense_tensors = manifest.get("dense_tensors", {})

    with open(dense_bin_p, "rb") as bin_f:
        # 1. Update NVFP4 tensors with scale_2
        nvfp4_count = 0
        for name, info in dense_tensors.items():
            if name.endswith(".weight"):
                s1_name = name + "_scale"
                s2_name = name + "_scale_2"
                if s1_name in dense_tensors and s2_name in dense_tensors:
                    info["dtype"] = "nvfp4"
                    s2_offset = dense_tensors[s2_name]["offset"]
                    bin_f.seek(s2_offset)
                    s2_val = struct.unpack("<f", bin_f.read(4))[0]
                    info["scale_2"] = s2_val
                    nvfp4_count += 1

        print(f"Updated {nvfp4_count} NVFP4 tensors in dense manifest.")

        # 2. Update FP8 Mamba tensors with scalar weight_scale
        fp8_mamba_count = 0
        for name, info in dense_tensors.items():
            if ("in_proj.weight" in name or "out_proj.weight" in name) and info.get("dtype") == "F8_E4M3":
                ws_name = name + "_scale"
                if ws_name in dense_tensors:
                    ws_offset = dense_tensors[ws_name]["offset"]
                    bin_f.seek(ws_offset)
                    ws_val = struct.unpack("<f", bin_f.read(4))[0]
                    info["scalar_scale"] = ws_val
                    fp8_mamba_count += 1

        print(f"Updated {fp8_mamba_count} FP8 Mamba projection tensors with scalar scales.")

    with open(manifest_p, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print(f"Successfully updated {manifest_p}!")

if __name__ == "__main__":
    update_manifest()
