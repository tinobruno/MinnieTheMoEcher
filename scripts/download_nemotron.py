#!/usr/bin/env python3
"""
download_nemotron.py — Download official NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4 weights.
Downloads all safetensors shards and metadata to a target directory with resume support.
"""

import os
import sys
import time
from huggingface_hub import snapshot_download

REPO_ID = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B-NVFP4"
TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "E:/dev/nemotron_raw"

def main():
    print("=================================================================")
    print(f"  Downloading Model: {REPO_ID}")
    print(f"  Target Directory:  {TARGET_DIR}")
    print("=================================================================\n")
    os.makedirs(TARGET_DIR, exist_ok=True)
    t0 = time.time()

    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        max_workers=8,
        allow_patterns=[
            "config.json",
            "model.safetensors.index.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "*.safetensors"
        ]
    )

    elapsed = time.time() - t0
    print("\n=================================================================")
    print(f"  DOWNLOAD COMPLETED in {elapsed:.1f}s ({elapsed/60.0:.2f} min)!")
    print(f"  Files resident at: {path}")
    print("=================================================================")

if __name__ == "__main__":
    main()
