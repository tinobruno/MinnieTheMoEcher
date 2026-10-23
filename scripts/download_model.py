#!/usr/bin/env python3
"""
download_model.py — Unified Model Downloader for MinnieTheMoEcher

Downloads optimized model packages from Hugging Face directly into models/
Supports resume, chunked download, and progress indication.

Usage:
    python scripts/download_model.py --model nemotron
    python scripts/download_model.py --model qwen_13g
    python scripts/download_model.py --model qwen
    python scripts/download_model.py --model deepseek
"""

import os
import sys
import time
import argparse
import urllib.request
import urllib.error

MODELS = {
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
    "qwen_13g": {
        "name": "Qwen 3.8 27B Vision (13GB GPU Mode)",
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

    # Get remote size
    req = urllib.request.Request(url, method="HEAD", headers=headers)
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
        # If file already exists locally, keep it rather than hard failing on offline/gated
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
    chunk_size = 4 * 1024 * 1024 # 4MB chunks
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

    os.rename(temp_path, out_path)
    print(f"[DONE] {filename} verified ({format_bytes(downloaded)})\n")
    return True

def main():
    parser = argparse.ArgumentParser(description="MinnieTheMoEcher Model Downloader")
    parser.add_argument(
        "--model", "-m",
        choices=list(MODELS.keys()),
        default="nemotron",
        help="Model package to download (default: nemotron)"
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
        if not download_file(cfg["repo"], f, dest_dir):
            print(f"\n[ERROR] Failed to download {f}")
            sys.exit(1)

    # Automatically install multimodal vision delegate by default if specified
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
