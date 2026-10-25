#!/usr/bin/env python3
"""
download_deepseek_v41.py — High-Speed Resumable Downloader for DeepSeek-V4.1-Flash

Downloads the official DeepSeek-V4.1-Flash-1017 checkpoint from Hugging Face
directly to the fast NVMe target directory on F:\\.
Supports multi-worker parallel streaming, automatic resume, and retry loops.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import snapshot_download

DEFAULT_REPO_ID = "deepseek-ai/DeepSeek-V4.1-Flash-1017"
DEFAULT_TARGET_DIR = "F:/models/deepseek_v41_raw"

def main():
    parser = argparse.ArgumentParser(description="Download DeepSeek-V4.1-Flash weights from Hugging Face")
    parser.add_argument("--repo", default=DEFAULT_REPO_ID, help=f"Hugging Face repository ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--dest", default=DEFAULT_TARGET_DIR, help=f"Destination directory (default: {DEFAULT_TARGET_DIR})")
    parser.add_argument("--workers", type=int, default=8, help="Number of concurrent download workers (default: 8)")
    parser.add_argument("--max-retries", type=int, default=50, help="Max retry attempts on network interruptions (default: 50)")
    args = parser.parse_args()

    repo_id = args.repo
    target_dir = os.path.abspath(args.dest)
    os.makedirs(target_dir, exist_ok=True)

    print("=" * 70)
    print(f"  DOWNLOADING DEEPSEEK-V4.1-FLASH (OFFICIAL LATEST)")
    print(f"  Repository:        {repo_id}")
    print(f"  Target Directory:  {target_dir}")
    print(f"  Concurrent Streams: {args.workers}")
    print("=" * 70, flush=True)

    t0 = time.time()
    max_retries = args.max_retries
    path = None

    for attempt in range(1, max_retries + 1):
        try:
            print(f"\n[INFO] Starting download phase (attempt {attempt}/{max_retries}, resume enabled)...", flush=True)
            path = snapshot_download(
                repo_id=repo_id,
                local_dir=target_dir,
                max_workers=args.workers,
                ignore_patterns=[
                    "*.pdf",
                    "assets/*",
                    "*.png",
                ],
                resume_download=True
            )
            print("\n[SUCCESS] All shards verified and downloaded completely!", flush=True)
            break
        except Exception as e:
            print(f"\n[WARN] Connection issue on attempt {attempt}: {type(e).__name__} ({e})", flush=True)
            if attempt == max_retries:
                print("\n[ERROR] Maximum retries reached. Please check connection and rerun.", flush=True)
                sys.exit(1)
            wait_s = min(30, 2 * attempt)
            print(f"[INFO] Retrying in {wait_s}s (local shards preserved)...", flush=True)
            time.sleep(wait_s)

    elapsed = time.time() - t0
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print(f"  DOWNLOAD COMPLETE in {hours}h {minutes}m {seconds}s!")
    print(f"  Weights stored at: {target_dir}")
    print("=" * 70, flush=True)

if __name__ == "__main__":
    main()
