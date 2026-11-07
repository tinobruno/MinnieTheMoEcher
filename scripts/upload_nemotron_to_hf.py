#!/usr/bin/env python3
"""
upload_nemotron_to_hf.py — Upload MinnieTheMoEcher Nemotron 3.5 Lightning package to Hugging Face.

Usage:
    python scripts/upload_nemotron_to_hf.py [--token <hf_token>] [--repo-id <repo_id>] [--model-dir <dir>]
"""

import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "TinoBruno/moecher-nemotron-3.5-lightning-vision"
DEFAULT_MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "nemotron_3.5_lightning"))

FILES_TO_UPLOAD = [
    "README.md",
    "moecher_manifest.json",
    "tokenizer.json",
    "expert_scales.bin",
    "attention_dense_layers.bin",
    "expert_weights.bin"
]

def format_size(bytes_val: int) -> str:
    if bytes_val >= 1024 ** 3:
        return f"{bytes_val / (1024 ** 3):.2f} GB"
    elif bytes_val >= 1024 ** 2:
        return f"{bytes_val / (1024 ** 2):.2f} MB"
    elif bytes_val >= 1024:
        return f"{bytes_val / 1024:.2f} KB"
    return f"{bytes_val} B"

def upload(repo_id: str, model_dir: Path, token: str = None, private: bool = False):
    api = HfApi(token=token)
    
    print("=================================================================", flush=True)
    print(f"  Target Hugging Face Repo: {repo_id}", flush=True)
    print(f"  Source Model Directory:   {model_dir}", flush=True)
    print("=================================================================\n", flush=True)

    print("[1/4] Checking Hugging Face authentication...", flush=True)
    try:
        user_info = api.whoami()
        print(f"  Authenticated as: {user_info.get('name')} ({user_info.get('fullname', '')})", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}", flush=True)
        print("\nPlease supply a valid Hugging Face write token:", flush=True)
        print("  - Command line: python scripts/upload_nemotron_to_hf.py --token <hf_token>", flush=True)
        print("  - Or environment variable: set HF_TOKEN=<hf_token>", flush=True)
        print("  - Or run: huggingface-cli login", flush=True)
        return False

    print(f"\n[2/4] Verifying local files in {model_dir}...", flush=True)
    total_bytes = 0
    for filename in FILES_TO_UPLOAD:
        local_path = model_dir / filename
        if not local_path.exists():
            print(f"  [ERROR] Required file not found: {local_path}", flush=True)
            return False
        sz = local_path.stat().st_size
        total_bytes += sz
        print(f"  - {filename:<28} ({format_size(sz)})", flush=True)
    print(f"  Total package payload: {format_size(total_bytes)}", flush=True)

    print(f"\n[3/4] Ensuring repository exists: {repo_id}...", flush=True)
    try:
        api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
        print(f"  Repository ready: https://huggingface.co/{repo_id}", flush=True)
    except Exception as e:
        print(f"  [WARN] create_repo warning/error: {e}", flush=True)

    try:
        existing_files = set(api.list_repo_files(repo_id=repo_id))
    except Exception:
        existing_files = set()

    print(f"\n[4/4] Uploading files to {repo_id}...", flush=True)
    for idx, filename in enumerate(FILES_TO_UPLOAD, 1):
        local_path = model_dir / filename
        sz = local_path.stat().st_size
        if filename in existing_files and filename != "README.md":
            print(f"  [{idx}/{len(FILES_TO_UPLOAD)}] [SKIP] {filename} already present on Hugging Face.", flush=True)
            continue
        print(f"  [{idx}/{len(FILES_TO_UPLOAD)}] Uploading {filename} ({format_size(sz)})...", flush=True)
        max_retries = 5
        for attempt in range(1, max_retries + 1):
            try:
                t0 = time.time()
                api.upload_file(
                    path_or_fileobj=str(local_path),
                    path_in_repo=filename,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Upload {filename} for {repo_id}"
                )
                elapsed = time.time() - t0
                speed_mb = (sz / (1024 * 1024)) / (elapsed if elapsed > 0 else 1)
                print(f"      -> Successfully uploaded {filename} in {elapsed:.1f}s ({speed_mb:.1f} MB/s)", flush=True)
                break
            except Exception as e:
                print(f"      [WARN] Attempt {attempt}/{max_retries} failed for {filename}: {e}", flush=True)
                if attempt == max_retries:
                    print(f"      [ERROR] Failed to upload {filename}: {e}", flush=True)
                    return False
                time.sleep(min(30, 2 ** attempt))

    print("\n=================================================================", flush=True)
    print("  ALL NEMOTRON PACKAGE FILES UPLOADED SUCCESSFULLY!", flush=True)
    print(f"  Model URL: https://huggingface.co/{repo_id}", flush=True)
    print("=================================================================\n", flush=True)
    return True

def main():
    parser = argparse.ArgumentParser(description="Upload Nemotron 3.5 Lightning package to Hugging Face")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help="HF Repo ID (e.g. TinoBruno/moecher-nemotron-3.5-lightning-vision)")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help="Local directory containing model files")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face API Write Token")
    parser.add_argument("--private", action="store_true", help="Set repository to private")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    model_dir = Path(args.model_dir)

    success = upload(repo_id=args.repo_id, model_dir=model_dir, token=token, private=args.private)
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
