#!/usr/bin/env python3
"""
upload_nemotron_bolt.py — Upload NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4) package to Hugging Face.
Supports automatic repo creation, model card generation, resumable multi-part upload,
and automatic 2-part sharding for weights > 50 GB (Hugging Face Git LFS limit).
"""

import os
import sys
import argparse
import time
from pathlib import Path
from huggingface_hub import HfApi, login

DEFAULT_REPO_ID = "TinoBruno/moecher-nemotron-3.5-bolt-nvfp4"
DEFAULT_MODEL_DIR = "models/nemotron_3.5_bolt"

FILES_TO_UPLOAD = [
    "README.md",
    "moecher_manifest_16gb.json",
    "moecher_manifest_24gb.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "expert_scales.bin",
    "expert_freq.bin",
    "expert_profile.json",
    "attention_dense_layers.bin",
    "expert_weights.bin"
]

MODEL_CARD_TEMPLATE = """---
language:
- en
license: other
license_name: nvidia-open-model-license
license_link: https://developer.download.nvidia.com/licenses/nvidia-open-model-license-agreement-june-2024.pdf
tags:
- moe
- mamba
- ssm
- nvfp4
- blackwell
- moecher
- agentic
- tool-calling
model_name: NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4-Moecher
base_model: nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-NVFP4
pipeline_tag: text-generation
---

# NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4) — MinnieTheMoEcher Package

This repository contains the optimized weights, manifests, and tokenizer for **NVIDIA Nemotron 3.5 Bolt (74B-A9B)** formatted for the **MinnieTheMoEcher** high-performance inference engine.

## Model Highlights
* **Architecture:** Hybrid Mamba-2 State-Space Model (SSM) + Grouped-Query Attention (GQA) + Sparse Latent Mixture-of-Experts (LatentMoE).
* **Scale:** 74 Billion total parameters with **9 Billion active parameters per token** (40 MoE layers, up to 496 routed experts per layer, `top_k = 9`).
* **Format:** Native **NVIDIA NVFP4** (4-bit floating point with FP8 block scales and FP32 global scaling factor) on Blackwell `sm_120a` Tensor Cores, with FP8 intermediate projections and BF16 embeddings/dense layers.
* **Dual Hardware Targets:**
  * `moecher_manifest_16gb.json`: Tuned for 16GB VRAM GPUs (GeForce RTX 4080 / 5060 Ti) using hybrid VRAM L1 cache, DRAM L2 pinned cache, and NVMe lookahead streaming.
  * `moecher_manifest_24gb.json`: Tuned for 24GB VRAM GPUs (GeForce RTX 3090 / 4090 / 5090) for 100% in-VRAM resident expert execution.

## File Inventory
| File | Size | Description |
| :--- | :--- | :--- |
| `attention_dense_layers.bin` | ~8.27 GB | BF16 embeddings, Mamba-2 SSM mixers, GQA attention heads, shared experts, and LM head |
| `expert_weights.bin.001` | ~27.28 GB | Routed MoE experts (Part 1/2, layers 0..19) |
| `expert_weights.bin.002` | ~27.28 GB | Routed MoE experts (Part 2/2, layers 20..39) |
| `expert_scales.bin` | ~155 KB | Per-expert global scale factors (`up_proj` and `down_proj`) |
| `moecher_manifest_16gb.json` | ~194 KB | Hardware profile for 16GB VRAM GPUs |
| `moecher_manifest_24gb.json` | ~194 KB | Hardware profile for 24GB VRAM GPUs |
| `tokenizer.json` | ~16.28 MB | BPE Tokenizer (131,072 vocabulary) |
| `tokenizer_config.json` | ~617 B | Tokenizer configuration and special tokens |
| `chat_template.jinja` | ~12.5 KB | Chat template with tool calling harness support |
| `expert_freq.bin` | ~174 KB | Pre-calibrated expert frequency profile for hot L1/L2 caching |
| `expert_profile.json` | ~10.2 MB | Diagnostic router statistics |

> **Note on expert_weights.bin:** Due to Hugging Face's 50 GB per-file limit, `expert_weights.bin` (~54.6 GB) is distributed in two parts (`.001` and `.002`). The Moecher installer automatically joins them, or you can join them manually:
> ```cmd
> copy /b expert_weights.bin.001 + expert_weights.bin.002 expert_weights.bin
> del expert_weights.bin.001 expert_weights.bin.002
> ```

## Running with MinnieTheMoEcher

### Quick Start
```bash
# 16GB VRAM GPU (RTX 4080 / 5060 Ti)
.\\moecher.exe -m models/nemotron_3.5_bolt/moecher_manifest_16gb.json --port 8001

# 24GB VRAM GPU (RTX 3090 / 4090 / 5090)
.\\moecher.exe -m models/nemotron_3.5_bolt/moecher_manifest_24gb.json --port 8001
```

Or using the launcher script:
```bash
.\\start_nemotron_bolt.bat 16gb
.\\start_nemotron_bolt.bat 24gb
```

## License
Subject to NVIDIA Open Model License Agreement.
"""

def generate_model_card(model_dir: Path):
    readme_path = model_dir / "README.md"
    print(f"Updating model card: {readme_path}")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.strip() + "\n")
    print("  [OK] README.md generated.")

def upload_file_with_retry(api: HfApi, repo_id: str, local_path: Path, filename: str, max_retries: int = 5):
    file_size_gb = local_path.stat().st_size / (1024 ** 3)
    print(f"\nUploading {filename} ({file_size_gb:.2f} GB)...")
    
    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {filename} ({file_size_gb:.2f} GB)"
            )
            elapsed = time.time() - t0
            speed_mb = (local_path.stat().st_size / (1024 * 1024)) / (elapsed if elapsed > 0 else 1)
            print(f"  [OK] Uploaded {filename} in {elapsed:.1f}s ({speed_mb:.1f} MB/s)")
            return True
        except Exception as e:
            print(f"  [WARN] Attempt {attempt}/{max_retries} failed for {filename}: {e}")
            if attempt == max_retries:
                print(f"  [ERROR] Failed to upload {filename} after {max_retries} attempts.")
                raise
            wait_s = min(30, 2 ** attempt)
            print(f"  Retrying in {wait_s}s...")
            time.sleep(wait_s)
    return False

def split_and_upload_large_file(api: HfApi, repo_id: str, src_path: Path, existing_files: set):
    total_size = src_path.stat().st_size
    half_size = total_size // 2
    # Ensure alignment to expert block size 2953216
    block_size = 2953216
    split_point = (half_size // block_size) * block_size
    
    part1_name = f"{src_path.name}.001"
    part2_name = f"{src_path.name}.002"
    part1_path = src_path.parent / part1_name
    part2_path = src_path.parent / part2_name

    chunk_size = 64 * 1024 * 1024  # 64 MB read/write buffer

    # Part 1
    if part1_name in existing_files:
        print(f"\n[SKIP] {part1_name} is already present on Hugging Face.")
    else:
        print(f"\nPreparing {part1_name} ({split_point / (1024**3):.2f} GB)...")
        with open(src_path, "rb") as fin, open(part1_path, "wb") as fout:
            bytes_left = split_point
            while bytes_left > 0:
                to_read = min(chunk_size, bytes_left)
                data = fin.read(to_read)
                if not data:
                    break
                fout.write(data)
                bytes_left -= len(data)
        print(f"  [OK] {part1_name} prepared locally.")
        upload_file_with_retry(api, repo_id, part1_path, part1_name)
        if part1_path.exists():
            part1_path.unlink()
            print(f"  [CLEANUP] Deleted temporary local {part1_name}.")

    # Part 2
    if part2_name in existing_files:
        print(f"\n[SKIP] {part2_name} is already present on Hugging Face.")
    else:
        part2_bytes = total_size - split_point
        print(f"\nPreparing {part2_name} ({part2_bytes / (1024**3):.2f} GB)...")
        with open(src_path, "rb") as fin, open(part2_path, "wb") as fout:
            fin.seek(split_point)
            bytes_left = part2_bytes
            while bytes_left > 0:
                to_read = min(chunk_size, bytes_left)
                data = fin.read(to_read)
                if not data:
                    break
                fout.write(data)
                bytes_left -= len(data)
        print(f"  [OK] {part2_name} prepared locally.")
        upload_file_with_retry(api, repo_id, part2_path, part2_name)
        if part2_path.exists():
            part2_path.unlink()
            print(f"  [CLEANUP] Deleted temporary local {part2_name}.")

def main():
    parser = argparse.ArgumentParser(description="Upload Nemotron 3.5 Bolt package to Hugging Face")
    parser.add_argument("--repo-id", type=str, default=DEFAULT_REPO_ID, help=f"Target Hugging Face repo ID (default: {DEFAULT_REPO_ID})")
    parser.add_argument("--model-dir", type=str, default=DEFAULT_MODEL_DIR, help=f"Local model directory (default: {DEFAULT_MODEL_DIR})")
    parser.add_argument("--token", type=str, default=None, help="Hugging Face Write Token (or set HF_TOKEN env var)")
    parser.add_argument("--private", action="store_true", help="Create repository as private")
    args = parser.parse_args()

    token = args.token or os.environ.get("HF_TOKEN")
    if not token:
        token_path = Path.home() / ".cache" / "huggingface" / "token"
        if token_path.exists():
            token = token_path.read_text(encoding="utf-8").strip()

    if not token:
        print("[ERROR] Hugging Face token required. Aborting.")
        sys.exit(1)

    api = HfApi(token=token)
    try:
        user_info = api.whoami()
        username = user_info.get("name") or user_info.get("fullname")
        print(f"\n[AUTH] Authenticated with Hugging Face as: {username}")
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}")
        sys.exit(1)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}")
        sys.exit(1)

    # Generate README.md model card
    generate_model_card(model_dir)

    # Ensure repository exists on Hugging Face
    print(f"\nChecking repository: {args.repo_id}...")
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)
        print(f"  [OK] Repository {args.repo_id} is ready (Private: {args.private}).")
    except Exception as e:
        print(f"  [ERROR] Failed to verify/create repository {args.repo_id}: {e}")
        sys.exit(1)

    try:
        existing_files = set(api.list_repo_files(repo_id=args.repo_id))
    except Exception as e:
        print(f"  [WARN] Could not list existing files: {e}")
        existing_files = set()

    t_start = time.time()

    # Upload files
    for filename in FILES_TO_UPLOAD:
        p = model_dir / filename
        if not p.exists():
            continue

        # Check if file > 50 GB (Hugging Face hard limit for single files)
        if p.stat().st_size > 50 * (1024 ** 3):
            print(f"\nFile {filename} ({p.stat().st_size / (1024**3):.2f} GB) exceeds Hugging Face 50GB limit. Sharding into 2 parts...")
            split_and_upload_large_file(api, args.repo_id, p, existing_files)
            continue

        if filename in existing_files:
            # Re-upload README.md if it changed, skip other large binaries if already present
            if filename != "README.md":
                print(f"[SKIP] {filename} is already uploaded on Hugging Face.")
                continue

        upload_file_with_retry(api, args.repo_id, p, filename)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  SUCCESSFULLY UPLOADED NEMOTRON 3.5 BOLT PACKAGE TO HUGGING FACE!")
    print(f"  Repository: https://huggingface.co/{args.repo_id}")
    print(f"  Total Time: {total_time/60.0:.2f} minutes")
    print("=" * 70 + "\n")

if __name__ == "__main__":
    main()
