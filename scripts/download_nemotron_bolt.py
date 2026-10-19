#!/usr/bin/env python3
"""
download_nemotron_bolt.py — Download NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4) from Hugging Face
and automatically prepare it for MinnieTheMoEcher.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "cloudnathan5/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4"
TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "E:/dev/nemotron_bolt_raw"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "models/nemotron_3.5_bolt"

def main():
    print("=================================================================")
    print(f"  Downloading Model: {REPO_ID}")
    print(f"  Target Directory:  {TARGET_DIR}")
    print(f"  Prepared Model:    {OUT_DIR}")
    print("=================================================================\n")
    os.makedirs(TARGET_DIR, exist_ok=True)
    t0 = time.time()

    # Download config, index, tokenizer and the 61 language model shards
    # (Shards 62 and 63 only contain unused vision encoder weights)
    print("Starting download from Hugging Face Hub (resume enabled)...")
    path = snapshot_download(
        repo_id=REPO_ID,
        local_dir=TARGET_DIR,
        max_workers=8,
        ignore_patterns=[
            "model-00062-of-00063.safetensors",
            "model-00063-of-00063.safetensors",
            "model-summary-idxs.safetensors",
            "*.png",
            "*.md"
        ]
    )

    elapsed = time.time() - t0
    print("\n=================================================================")
    print(f"  DOWNLOAD COMPLETED in {elapsed:.1f}s ({elapsed/60.0:.2f} min)!")
    print(f"  Raw files resident at: {path}")
    print("=================================================================\n")

    print("Running zero-copy preparation script (prepare_nemotron_bolt.py)...")
    prep_script = Path(__file__).parent / "prepare_nemotron_bolt.py"
    cmd = [sys.executable, str(prep_script), TARGET_DIR, OUT_DIR]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n=================================================================")
        print("  NEMOTRON 3.5 BOLT READY FOR INFERENCE!")
        print("  To launch:")
        print("    .\\start_nemotron_bolt.bat 16gb    (16GB VRAM GPUs)")
        print("    .\\start_nemotron_bolt.bat 24gb    (24GB VRAM GPUs)")
        print("=================================================================\n")
    else:
        print(f"\n[ERROR] Preparation script exited with code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
