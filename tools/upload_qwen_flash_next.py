#!/usr/bin/env python3
"""
upload_qwen_flash_next.py — Upload Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M) package to Hugging Face.
"""

import os
import sys
import time
import argparse
from pathlib import Path
from huggingface_hub import HfApi

DEFAULT_REPO_ID = "TinoBruno/moecher-qwen-3.8-flash-next"
DEFAULT_MODEL_DIR = "models/qwen3.8_flash_next"

FILES_TO_UPLOAD = [
    "README.md",
    "moecher_manifest.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "config.json",
    "generation_config.json",
    "rco-allocation.txt",
    "imatrix.dat.counts.json",
    "expert_freq.bin",
    "mmproj-Qwen3.8-Flash-Next-BF16.gguf",
    "resident.bin",
    "experts.bin",
    "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf",
]

MODEL_CARD_TEMPLATE = """---
language:
- en
- zh
license: apache-2.0
tags:
- moe
- qwen
- deltanet
- ssm
- ple
- moecher
- agentic
- vision
- tool-calling
model_name: Qwen-3.8-Flash-Next-GSQ-RCO-IQ1_M-Moecher
pipeline_tag: image-text-to-text
---

# Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M) — MinnieTheMoEcher Package

This repository contains the optimized binary weights, manifest, multimodal vision projector (`mmproj`), and Per-Layer Embedding (PLE) N-gram lookup table for **Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M)** formatted for the **[MinnieTheMoEcher](https://github.com/tinobruno/MinnieTheMoEcher)** bare-metal agentic AI runtime.

## Model Highlights
* **Architecture (`qwen4exp`):** 48-layer Hybrid Gated DeltaNet SSM (36 layers) + Full Grouped-Query Attention (12 layers) + Hyper-Connections (`hc_mult = 4`, $D_{hc} = 10,240$) + Per-Layer Embedding (PLE) 320M N-gram Table + Sparse Mixture-of-Experts (256 routed experts per layer, `top_k = 10`, plus 1 gated shared expert).
* **Mixed-Precision RCO Quantization (`GSQ-RCO-IQ1_M`):**
  * `resident.bin` (~4.15 GB): High-precision `Q6_K` / `Q5_K` / `Q4_K` / `IQ4_XS` / `BF16` dense attention, DeltaNet SSM projections, shared experts, Hyper-Connections, PLE projections, and output head—kept 100% resident in GPU VRAM.
  * `experts.bin` (~23.42 GB): 48 layers × 256 routed MoE experts with heterogeneous per-layer RCO quantization (`IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `Q2_K`, `Q3_K`, `Q4_K`, `IQ4_NL`, `IQ4_XS`), streamed via Pinned Host DRAM L2 cache and asynchronous double-buffered DMA.
  * `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` (~26.82 GB): `IQ4_NL` Per-Layer Embedding (PLE) N-gram lookup table (`[160, 320001536]`) mapped zero-copy from host memory.
  * `mmproj-Qwen3.8-Flash-Next-BF16.gguf` (~865.5 MB): Native BF16 multimodal vision tower & projector.
* **Consumer 16GB GPU Ready:** Runs smoothly on a 16GB VRAM GPU (e.g., GeForce RTX 5060 Ti 16GB + 32GB/64GB DDR5 RAM) at **~113 tok/s prefill** and **~20.5 tok/s decode** with 0% CPU compute bottleneck.

## File Inventory
| File | Size | Description |
| :--- | :--- | :--- |
| `moecher_manifest.json` | ~220 KB | MinnieTheMoEcher tensor manifest & per-expert heterogeneous RCO layout |
| `resident.bin` | ~4.15 GB | VRAM-resident dense attention, DeltaNet SSM, Hyper-Connections, PLE & shared experts |
| `experts.bin` | ~23.42 GB | 12,288 routed MoE experts (heterogeneous GSQ-RCO-IQ1_M) |
| `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` | ~26.82 GB | PLE N-gram embedding lookup table (`ple_ngram_table.weight`, `IQ4_NL`) |
| `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | ~865.5 MB | Multimodal vision encoder & projector (BF16) |
| `tokenizer.json` | ~12.2 MB | BPE Tokenizer (248,320 vocabulary) |
| `expert_freq.bin` | ~48 KB | Calibrated MoE expert activation frequency profile for L1/L2 cache warming |
| `rco-allocation.txt` | ~2.62 MB | Per-tensor Rate-Constrained Optimization (RCO) quantization allocation log |

## Quick Start with MinnieTheMoEcher

```bash
# Download automatically via MinnieTheMoEcher downloader:
python scripts/download_model.py --model qwen_flash_next

# Launch the runtime:
./moecher -m models/qwen3.8_flash_next/moecher_manifest.json --port 8001
```
"""

def generate_model_card(model_dir: Path):
    readme_path = model_dir / "README.md"
    print(f"Generating model card: {readme_path}")
    with open(readme_path, "w", encoding="utf-8") as f:
        f.write(MODEL_CARD_TEMPLATE.strip() + "\n")
    print("  [OK] README.md generated.")

def upload_file_with_retry(api: HfApi, repo_id: str, local_path: Path, filename: str, max_retries: int = 5):
    size_bytes = local_path.stat().st_size
    size_gb = size_bytes / (1024 ** 3)
    size_mb = size_bytes / (1024 ** 2)
    size_label = f"{size_gb:.2f} GB" if size_gb >= 1.0 else f"{size_mb:.2f} MB"
    print(f"\nUploading {filename} ({size_label})...", flush=True)

    for attempt in range(1, max_retries + 1):
        try:
            t0 = time.time()
            api.upload_file(
                path_or_fileobj=str(local_path),
                path_in_repo=filename,
                repo_id=repo_id,
                repo_type="model",
                commit_message=f"Upload {filename} ({size_label})"
            )
            elapsed = time.time() - t0
            speed_mb = size_mb / (elapsed if elapsed > 0 else 1)
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
    parser = argparse.ArgumentParser(description="Upload Qwen 3.8 Flash-Next package to Hugging Face")
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

    api = HfApi(token=token)
    user_info = api.whoami()
    username = user_info.get("name") or user_info.get("fullname")
    print(f"[AUTH] Authenticated with Hugging Face as: {username}", flush=True)

    model_dir = Path(args.model_dir).resolve()
    if not model_dir.exists():
        print(f"[ERROR] Model directory not found: {model_dir}", flush=True)
        sys.exit(1)

    generate_model_card(model_dir)

    print(f"\nChecking repository: {args.repo_id}...", flush=True)
    api.create_repo(repo_id=args.repo_id, repo_type="model", exist_ok=True, private=args.private)
    print(f"  [OK] Repository {args.repo_id} is ready (Private: {args.private}).", flush=True)

    try:
        existing_files = set(api.list_repo_files(repo_id=args.repo_id))
    except Exception:
        existing_files = set()

    t_start = time.time()
    for filename in FILES_TO_UPLOAD:
        p = model_dir / filename
        if not p.exists():
            print(f"[WARN] Skipping {filename} (not found at {p})", flush=True)
            continue
        if filename in existing_files and filename != "README.md":
            print(f"[SKIP] {filename} is already uploaded on Hugging Face.", flush=True)
            continue
        upload_file_with_retry(api, args.repo_id, p, filename)

    total_time = time.time() - t_start
    print("\n" + "=" * 70, flush=True)
    print(f"  SUCCESSFULLY UPLOADED QWEN 3.8 FLASH-NEXT PACKAGE TO HUGGING FACE!", flush=True)
    print(f"  Repository: https://huggingface.co/{args.repo_id}", flush=True)
    print(f"  Total Time: {total_time/60.0:.2f} minutes", flush=True)
    print("=" * 70 + "\n", flush=True)

if __name__ == "__main__":
    main()

