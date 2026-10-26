#!/usr/bin/env python3
"""
download_qwen3_8_turbo.py — Resumable downloader for DavidAU's unquantized Qwen3.8-27B-TURBO model.
Downloads all safetensors shards, tokenizer, and config files with automatic retry and exponential backoff.

Usage:
    python scripts/download_qwen3_8_turbo.py [TARGET_DIR]
"""

import os
import sys
import time
from pathlib import Path
from huggingface_hub import snapshot_download

REPO_ID = "DavidAU/Qwen3.8-27B-TURBO-Fable-Cold-Fusion-735-882-Heretic-Uncensored-NM-DAU"
DEFAULT_TARGET = r"E:\models\qwen3_8_turbo_raw"

def main():
    target_dir = sys.argv[1] if len(sys.argv) > 1 else DEFAULT_TARGET
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 70)
    print(f"  Downloading Model: {REPO_ID}")
    print(f"  Destination Path:  {target_dir}")
    print("=" * 70)

    t0 = time.time()
    max_retries = 25
    success = False

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[Attempt {attempt}/{max_retries}] Starting / Resuming snapshot download...")
            path = snapshot_download(
                repo_id=REPO_ID,
                local_dir=target_dir,
                max_workers=4,
                allow_patterns=[
                    "*.safetensors",
                    "*.json",
                    "*.jinja",
                    "*.txt",
                    "tokenizer*",
                ],
                ignore_patterns=[
                    "*.gif",
                    "*.png",
                    "*.jpg",
                ]
            )
            success = True
            print("\n[SUCCESS] All files downloaded and verified!")
            break
        except Exception as e:
            print(f"\n[WARN] Connection interrupted on attempt {attempt}: {type(e).__name__} ({e})")
            if attempt == max_retries:
                print("\n[ERROR] Maximum retries reached. Please verify internet connection.")
                raise
            wait_s = min(30, 2 * attempt)
            print(f"Retrying in {wait_s}s (local shards preserved, resuming via HTTP range)...")
            time.sleep(wait_s)

    elapsed = time.time() - t0
    if success:
        print("=" * 70)
        print(f"  DOWNLOAD COMPLETE in {elapsed:.1f}s ({elapsed / 60.0:.2f} min)!")
        print(f"  Files saved at: {target_dir}")
        print("=" * 70)

if __name__ == "__main__":
    main()
