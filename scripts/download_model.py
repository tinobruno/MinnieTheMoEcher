#!/usr/bin/env python3
"""
download_model.py — Unified Model Downloader for MinnieTheMoEcher

Downloads optimized model packages from Hugging Face directly into models/
Supports resume, chunked download, and progress indication.

Usage:
    python scripts/download_model.py --model qwen_13g
    python scripts/download_model.py --model nemotron
    python scripts/download_model.py --model nemotron_bolt
    python scripts/download_model.py --model qwen
    python scripts/download_model.py --model deepseek
    python scripts/download_model.py --model deepseek_q4
"""

import os
import sys
import time
import shutil
import argparse
import urllib.request
import urllib.error

MODELS = {
    "qwen_flash_next": {
        "name": "Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M — 16GB/24GB GPU Hybrid)",
        "repo": "TinoBruno/moecher-qwen-3.8-flash-next",
        "dir": "qwen3.8_flash_next",
        "files": [
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
        ],
    },
    "qwen_13g": {
        "name": "Qwen 3.8 27B Vision (13GB/16GB GPU & Apple Silicon Mode)",
        "repo": "TinoBruno/moecher-qwen-3.8-27b-vision-13g",
        "dir": "qwen3_8_27b_vision_13g",
        "files": [
            "moecher_manifest.json",
            "tokenizer.json",
            "draft_vocab_ids.bin",
            "draft_vocab_ids.json",
            "draft_vocab_ids_mapping.json",
            "draft_vocab_ids_reverse.bin",
            "draft_lm_head_int8.bin",
            "draft_lm_head_int8_bf16.bin",
            "attention_dense_layers.bin",
        ],
    },
    "nemotron": {
        "name": "NVIDIA Nemotron 3.5 Lightning (30B-A3B NVFP4) + FrankensTin Vision",
        "repo": "TinoBruno/moecher-nemotron-3.5-lightning-vision",
        "dir": "nemotron_3.5_lightning",
        "files": [
            "moecher_manifest.json",
            "tokenizer.json",
            "expert_scales.bin",
            "attention_dense_layers.bin",
            "expert_weights.bin",
        ],
        "delegate": {
            "name": "FrankensTin Vision Delegate (Qwen2.5-VL 3B ViT)",
            "repo": "TinoBruno/frankenstin-vision-delegate",
            "dir": "frankenstin/vision",
            "files": [
                "moecher_manifest.json",
                "tokenizer.json",
                "tokenizer_config.json",
                "preprocessor_config.json",
                "chat_template.json",
                "merges.txt",
                "vocab.json",
                "attention_dense_layers.bin",
            ],
        },
    },
    "nemotron_bolt": {
        "name": "NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4 — 16GB Hybrid / 24GB Resident)",
        "repo": "TinoBruno/moecher-nemotron-3.5-bolt-nvfp4",
        "dir": "nemotron_3.5_bolt",
        "files": [
            "moecher_manifest_16gb.json",
            "moecher_manifest_24gb.json",
            "tokenizer.json",
            "tokenizer_config.json",
            "chat_template.jinja",
            "expert_scales.bin",
            "expert_freq.bin",
            "expert_profile.json",
            "attention_dense_layers.bin",
            "expert_weights.bin.001",
            "expert_weights.bin.002",
        ],
    },
    "qwen": {
        "name": "Qwen 3.8 27B INT4 with MTP Speculative Decoding",
        "repo": "TinoBruno/moecher-qwen-3.8-27b-q4",
        "dir": "qwen3_8_27b_q4",
        "files": [
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers_q4.bin",
            "draft_vocab_ids.bin",
            "draft_lm_head_int8_bf16.bin",
        ],
    },
    "deepseek": {
        "name": "DeepSeek V4 Flash IQ2 Standard",
        "repo": "TinoBruno/moecher-deepseek-v4-flash-iq2",
        "dir": "deepseek_v4_flash_iq2",
        "files": [
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers.bin",
            "moe_experts_iq2.bin",
        ],
    },
    "deepseek_q4": {
        "name": "DeepSeek V4 Flash Q4 (8GB GPU Mode)",
        "repo": "TinoBruno/moecher-deepseek-v4-flash-q4",
        "dir": "deepseek_v4_flash_q4",
        "files": [
            "moecher_manifest.json",
            "tokenizer.json",
            "attention_dense_layers_q4.bin",
            "moe_experts_iq2.bin",
        ],
    },
}

def format_bytes(n):
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"
    elif n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"

def get_auth_token():
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
    if not token:
        try:
            from huggingface_hub import get_token
            token = get_token()
        except Exception:
            pass
    return token

def download_file(repo, filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, filename)
    url = f"https://huggingface.co/{repo}/resolve/main/{filename}"

    headers = {"User-Agent": "MoecherDownloader/1.0"}
    token = get_auth_token()
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        if os.path.exists(out_path) and os.path.getsize(out_path) > 0:
            print(f"[FOUND] {filename} already present locally ({format_bytes(os.path.getsize(out_path))})")
            return True
        print(f"Error fetching metadata for {filename}: {e}")
        return False

    existing_size = 0
    if os.path.exists(out_path):
        existing_size = os.path.getsize(out_path)
        if total_size > 0 and existing_size == total_size:
            print(f"[SKIP] {filename} already fully downloaded ({format_bytes(existing_size)})")
            return True

    temp_path = out_path + ".part"
    mode = "wb"
    downloaded = 0
    if os.path.exists(temp_path):
        downloaded = os.path.getsize(temp_path)
        if downloaded < total_size:
            headers["Range"] = f"bytes={downloaded}-"
            mode = "ab"
        else:
            downloaded = 0

    print(f"Downloading {filename} ({format_bytes(total_size)})...")
    req = urllib.request.Request(url, headers=headers)
    chunk_size = 4 * 1024 * 1024
    start_time = time.time()
    last_print = start_time

    try:
        with urllib.request.urlopen(req) as resp, open(temp_path, mode) as f:
            while True:
                chunk = resp.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                downloaded += len(chunk)
                now = time.time()
                if now - last_print >= 2.0 or downloaded == total_size:
                    elapsed = now - start_time
                    speed = (downloaded - (0 if mode == "wb" else os.path.getsize(temp_path) - downloaded)) / max(elapsed, 0.001)
                    pct = (downloaded / total_size * 100.0) if total_size > 0 else 0.0
                    print(f"  {filename}: {format_bytes(downloaded)} / {format_bytes(total_size)} ({pct:.1f}%) @ {format_bytes(speed)}/s")
                    last_print = now
    except Exception as e:
        print(f"\nError downloading {filename}: {e}")
        return False

    os.replace(temp_path, out_path)
    print(f"[DONE] {filename} verified ({format_bytes(downloaded)})\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="MinnieTheMoEcher Model Downloader")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default="qwen_13g",
        help="Model package to download (default: qwen_13g)"
    )
    parser.add_argument(
        "--dest-dir", "-d",
        type=str,
        default=None,
        help="Override destination directory"
    )
    parser.add_argument(
        "--skip-vision",
        action="store_true",
        help="Skip downloading multimodal vision delegate if configured"
    )
    args = parser.parse_args()

    cfg = MODELS[args.model]
    base_models_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models"))
    dest_dir = args.dest_dir or os.path.join(base_models_dir, cfg["dir"])

    print("=================================================================")
    print(f"  Model:            {cfg['name']}")
    print(f"  Hugging Face:     {cfg['repo']}")
    print(f"  Target Directory: {dest_dir}")
    print("=================================================================\n")

    for f in cfg["files"]:
        if f in ("expert_weights.bin.001", "expert_weights.bin.002"):
            assembled = os.path.join(dest_dir, "expert_weights.bin")
            if os.path.exists(assembled) and os.path.getsize(assembled) > 0:
                print(f"[SKIP] expert_weights.bin already assembled ({format_bytes(os.path.getsize(assembled))})")
                continue
        if not download_file(cfg["repo"], f, dest_dir):
            print(f"\n[ERROR] Failed to download {f}")
            sys.exit(1)

    if args.model == "nemotron_bolt":
        p1 = os.path.join(dest_dir, "expert_weights.bin.001")
        p2 = os.path.join(dest_dir, "expert_weights.bin.002")
        target = os.path.join(dest_dir, "expert_weights.bin")
        if os.path.exists(p1) and os.path.exists(p2) and not os.path.exists(target):
            print("Assembling expert_weights.bin from split parts...")
            with open(target, "wb") as out_f:
                for part in (p1, p2):
                    with open(part, "rb") as in_f:
                        shutil.copyfileobj(in_f, out_f, length=16 * 1024 * 1024)
            os.remove(p1)
            os.remove(p2)
            print(f"[DONE] expert_weights.bin assembled ({format_bytes(os.path.getsize(target))})\n")

    if "delegate" in cfg and not args.skip_vision:
        del_cfg = cfg["delegate"]
        del_dest = os.path.join(base_models_dir, del_cfg["dir"])
        print("\n=================================================================")
        print(f"  Vision Delegate:  {del_cfg['name']}")
        print(f"  Hugging Face:     {del_cfg['repo']}")
        print(f"  Target Directory: {del_dest}")
        print("=================================================================\n")
        for f in del_cfg["files"]:
            if not download_file(del_cfg["repo"], f, del_dest):
                print(f"\n[ERROR] Failed to download vision delegate file {f}")
                sys.exit(1)

    print("=================================================================")
    print(f"  ALL FILES DOWNLOADED AND VERIFIED FOR {cfg['name']}!")
    print("=================================================================")

if __name__ == "__main__":
    main()
