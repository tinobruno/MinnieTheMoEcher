#!/usr/bin/env python3
"""
upload_nemotron_bolt_24gb.py — Upload NVIDIA Nemotron 3.5 Bolt 24GB (74B-A9B IQ2_XXS) package to Hugging Face.
Supports automatic repo creation, model card generation, and resumable upload with retries.
"""

import os
import sys
import argparse
import time
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "TinoBruno/moecher-nemotron-3.5-bolt-24gb"
DEFAULT_MODEL_DIR = "models/nemotron_3.5_bolt_24gb"

FILES_TO_UPLOAD = [
    "README.md",
    "moecher_manifest.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "chat_template.jinja",
    "expert_scales.bin",
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
- iq2
- blackwell
- moecher
- agentic
- tool-calling
model_name: NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB-Moecher
base_model: nvidia/NVIDIA-Nemotron-3.5-Bolt-74B-A9B-24GB
pipeline_tag: text-generation
---

# NVIDIA Nemotron 3.5 Bolt 24GB (74B-A9B IQ2_XXS) — MinnieTheMoEcher Package

This repository contains the pre-aligned binary weights, manifest, and tokenizer for **NVIDIA Nemotron 3.5 Bolt 24GB (74B-A9B)** formatted for the **MinnieTheMoEcher** bare-metal C++/CUDA inference engine.

## Model Highlights
* **Architecture:** Hybrid Mamba-2 State-Space Model (SSM) + Grouped-Query Attention (GQA) + Sparse Latent Mixture-of-Experts (`nemotron_h_puzzle` / LatentMoE).
* **Scale:** 74 Billion total parameters with **9 Billion active parameters per token** (88 total layers: 40 MoE layers with 496 routed experts per layer, `top_k = 9`).
* **Format:** Page-aligned (4096-byte) **IQ2_XXS** routed LatentMoE experts (~1.29 MB per expert block) paired with FP8/BF16 resident Mamba-2 SSM, GQA attention, bottlenecks, shared experts, and LM head.
* **100% Memory-Resident Consumer Execution:**
  * **16GB VRAM + 32GB Host RAM (e.g. RTX 5060 Ti 16GB / RTX 4080):** 6.40 GB L1 VRAM + 13.50 GB pinned host DRAM L2 with asynchronous double-buffered DMA prefetching (**21.54 tok/s prefill, 13.51 tok/s decode** with 0% CPU load — **3.86x faster** than `llama.cpp` MoE cache).
  * **24GB+ VRAM GPUs (e.g. RTX 3090 / 4090 / 5090):** Ultra-fast in-memory execution with zero NVMe SSD reads during generation.

## File Inventory
| File | Size | Description |
| :--- | :--- | :--- |
| `attention_dense_layers.bin` | ~4.98 GiB | Resident weights (Embeddings, Mamba-2 SSM mixers, GQA attention, LatentMoE bottlenecks, shared experts, LM head) |
| `expert_weights.bin` | ~24.98 GiB | 19,840 page-aligned routed LatentMoE experts in IQ2_XXS format |
| `expert_scales.bin` | ~155 KB | Per-expert FP32 scale factors (`up_proj` and `down_proj`) |
| `moecher_manifest.json` | ~160 KB | Complete architecture configuration and byte-exact tensor layout |
| `tokenizer.json` | ~16.29 MB | BPE Tokenizer (131,072 vocabulary) |
| `tokenizer_config.json` | ~617 B | Tokenizer configuration and special tokens |
| `chat_template.jinja` | ~12.2 KB | Official chat template with reasoning and tool-calling support |

## Running with MinnieTheMoEcher

### Quick Start
```bash
# Launch server on port 8001 (Windows)
.\\moecher.exe -m models/nemotron_3.5_bolt_24gb/moecher_manifest.json --port 8001

# Or use the one-click launcher:
.\\start_nemotron_bolt_24gb.bat
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
    print(f"\nUploading {filename} ({file_size_gb:.2f} GiB)...", flush=True)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {filename} ({file_size_gb:.2f} GiB)"
            )
            elapsed = time.time() - t0
            speed_mb = (local_path.stat().st_size / (1024 * 1024)) / (elapsed if elapsed > 0 else 1)
            print(f"  [OK] Uploaded {filename} in {elapsed:.1f}s ({speed_mb:.1f} MB/s)", flush=True)
            return True
        except Exception as e:
            print(f"  [WARN] Attempt {attempt}/{max_retries} failed for {filename}: {e}", flush=True)
            if attempt == max_retries:
                print(f"  [ERROR] Failed to upload {filename} after {max_retries} attempts.", flush=True)
                raise
            wait_s = min(30, 2 ** attempt)
            print(f"  Retrying in {wait_s}s...", flush=True)
            time.sleep(wait_s)
    return False

def main():
    parser = argparse.ArgumentParser(description="Upload Nemotron 3.5 Bolt 24GB package to Hugging Face")
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
        print(f"\n[AUTH] Authenticated with Hugging Face as: {username}", flush=True)
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {e}", flush=True)
        sys.exit(1)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}", flush=True)
        sys.exit(1)

    generate_model_card(model_dir)

    print(f"\nChecking repository: {args.repo_id}...", flush=True)
    try:
        api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)
        print(f"  [OK] Repository {args.repo_id} is ready (Private: {args.private}).", flush=True)
    except Exception as e:
        print(f"  [ERROR] Failed to verify/create repository {args.repo_id}: {e}", flush=True)
        sys.exit(1)

    try:
        existing_files = set(api.list_repo_files(repo_id=args.repo_id))
    except Exception as e:
        print(f"  [WARN] Could not list existing files: {e}", flush=True)
        existing_files = set()

    t_start = time.time()

    for filename in FILES_TO_UPLOAD:
        p = model_dir / filename
        if not p.exists():
            print(f"  [WARN] Local file missing, skipping: {p}", flush=True)
            continue

        if filename in existing_files and filename != "README.md":
            print(f"[SKIP] {filename} is already uploaded on Hugging Face.", flush=True)
            continue

        upload_file_with_retry(api, args.repo_id, p, filename)

    total_time = time.time() - t_start
    print("\n" + "=" * 70)
    print(f"  SUCCESSFULLY UPLOADED NEMOTRON 3.5 BOLT 24GB PACKAGE TO HUGGING FACE!")
    print(f"  Repository: https://huggingface.co/{args.repo_id}")
    print(f"  Total Time: {total_time/60.0:.2f} minutes")
    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()

