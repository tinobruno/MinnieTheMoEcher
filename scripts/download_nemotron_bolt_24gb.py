#!/usr/bin/env python3
"""
download_nemotron_bolt_24gb.py — Download NVIDIA Nemotron 3.5 Bolt (74B-A9B 24GB) from Hugging Face
and automatically prepare it for MinnieTheMoEcher. Includes automatic retry and resume.
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB"
TARGET_DIR = sys.argv[1] if len(sys.argv) > 1 else "E:/dev/nemotron_bolt_24gb_raw"
OUT_DIR = sys.argv[2] if len(sys.argv) > 2 else "models/nemotron_3.5_bolt_24gb"

def main():
    print("=================================================================")
    print(f"  Downloading Model: {REPO_ID}")
    print(f"  Target Directory:  {TARGET_DIR}")
    print(f"  Prepared Model:    {OUT_DIR}")
    print("=================================================================\n")
    os.makedirs(TARGET_DIR, exist_ok=True)
    t0 = time.time()

    # Download config, index, tokenizer and the 89 language model shards
    # (Shard 90 only contains unused vision encoder weights)
    path = None
    max_retries = 20
    for attempt in range(1, max_retries + 1):
        try:
            print(f"Starting download attempt {attempt}/{max_retries} (resume enabled)...")
            path = snapshot_download(
                repo_id=REPO_ID,
                local_dir=TARGET_DIR,
                max_workers=4,
                ignore_patterns=[
                    "model-00090-of-00090.safetensors",
                    "*.png",
                    "*.md"
                ]
            )
            print("Download phase verified & completed!")
            break
        except Exception as e:
            print(f"\n[WARN] Connection issue on attempt {attempt}: {type(e).__name__} ({e})")
            if attempt == max_retries:
                print("\n[ERROR] Maximum retries reached. Please check network connection.")
                raise
            wait_s = min(30, 2 * attempt)
            print(f"Retrying in {wait_s}s (local shards preserved, resuming remaining files)...\n")
            time.sleep(wait_s)

    elapsed = time.time() - t0
    print("\n=================================================================")
    print(f"  DOWNLOAD COMPLETED in {elapsed:.1f}s ({elapsed/60.0:.2f} min)!")
    print(f"  Raw files resident at: {path}")
    print("=================================================================\n")

    print("Running zero-copy preparation script (prepare_nemotron_bolt_24gb.py)...")
    prep_script = Path(__file__).parent / "prepare_nemotron_bolt_24gb.py"
    cmd = [sys.executable, str(prep_script), TARGET_DIR, OUT_DIR]
    res = subprocess.run(cmd)
    if res.returncode == 0:
        print("\n=================================================================")
        print("  NEMOTRON 3.5 BOLT 24GB READY FOR INFERENCE!")
        print("  To launch:")
        print("    .\\start_nemotron_bolt_24gb.bat")
        print("=================================================================\n")
    else:
        print(f"\n[ERROR] Preparation script exited with code {res.returncode}")
        sys.exit(res.returncode)

if __name__ == "__main__":
    main()
