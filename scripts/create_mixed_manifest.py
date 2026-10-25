#!/usr/bin/env python3
import json
from pathlib import Path

def main():
    src = Path("F:/Moecher/models/deepseek_v41_nvfp4/moecher_manifest.json")
    dst = Path("F:/Moecher/models/deepseek_v41_nvfp4/moecher_manifest_mixed.json")

    print(f"Reading {src}...")
    with open(src, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    manifest["model_config"]["expert_dtype"] = "mixed_nvfp4_iq2"
    manifest["expert_bins"] = {
        "hot": {
            "path": "moe_experts_hot.bin",
            "block_size": 18800640
        },
        "cold": {
            "path": "moe_experts_iq2.bin",
            "block_size": 9953280
        }
    }
    manifest["expert_map_file"] = "mixed_expert_map.bin"

    print(f"Writing {dst}...")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("[SUCCESS] Created moecher_manifest_mixed.json!")

if __name__ == "__main__":
    main()
