#!/usr/bin/env python3
import os
import sys
import time
import urllib.request
import urllib.error

REPO = "TinoBruno/moecher-qwen-3.8-27b-vision-13g"
DEST_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "models", "qwen3_8_27b_vision_13g"))

FILES = [
    "moecher_manifest.json",
    "tokenizer.json",
    "draft_vocab_ids.bin",
    "draft_vocab_ids.json",
    "draft_vocab_ids_mapping.json",
    "draft_vocab_ids_reverse.bin",
    "draft_lm_head_int8.bin",
    "draft_lm_head_int8_bf16.bin",
    "attention_dense_layers.bin",
]

def format_bytes(n):
    if n >= 1024 * 1024 * 1024:
        return f"{n / (1024 * 1024 * 1024):.2f} GB"
    elif n >= 1024 * 1024:
        return f"{n / (1024 * 1024):.1f} MB"
    elif n >= 1024:
        return f"{n / 1024:.1f} KB"
    return f"{n} B"

def download_file(filename, target_dir):
    os.makedirs(target_dir, exist_ok=True)
    out_path = os.path.join(target_dir, filename)
    url = f"https://huggingface.co/{REPO}/resolve/main/{filename}"

    # Get remote size
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": "MoecherDownloader/1.0"})
    try:
        with urllib.request.urlopen(req) as resp:
            total_size = int(resp.headers.get("Content-Length", 0))
    except Exception as e:
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
    headers = {"User-Agent": "MoecherDownloader/1.0"}
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
    print("=================================================================")
    print(f"  Downloading Model: {REPO}")
    print(f"  Target Directory:  {DEST_DIR}")
    print("=================================================================\n")

    for f in FILES:
        if not download_file(f, DEST_DIR):
            print(f"\nFailed on {f}")
            sys.exit(1)

    print("\n=================================================================")
    print("  ALL MODEL FILES DOWNLOADED AND READY!")
    print("=================================================================")

if __name__ == "__main__":
    main()
