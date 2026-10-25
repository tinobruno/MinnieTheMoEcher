#!/usr/bin/env python3
"""
auto_pipeline_v41.py — Autonomous Pipeline for DeepSeek-V4.1-Flash

Monitors the background Hugging Face download, triggers the converter and Direct-I/O packer,
launches the Moecher engine with RTX 6000 3-level caching, and performs an end-to-end smoke test.
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess
from pathlib import Path

RAW_DIR = Path("F:/models/deepseek_v41_raw")
OUT_DIR = Path("F:/Moecher/models/deepseek_v41_nvfp4")
EXPECTED_SHARDS = 48
SERVER_PORT = 8001

def get_downloaded_shards():
    if not RAW_DIR.exists():
        return []
    return [f for f in os.listdir(RAW_DIR) if f.startswith("model-") and f.endswith(".safetensors")]

def get_downloaded_size_gb():
    if not RAW_DIR.exists():
        return 0.0
    total = sum(os.path.getsize(os.path.join(r, f)) for r, d, fs in os.walk(RAW_DIR) for f in fs)
    return total / (1024 ** 3)

def wait_for_download():
    print("=" * 70)
    print("  [PIPELINE STEP 1] Monitoring DeepSeek-V4.1-Flash Download")
    print(f"  Target: {EXPECTED_SHARDS} Safetensors Shards (~475 GB)")
    print("=" * 70, flush=True)

    t0 = time.time()
    last_count = 0
    while True:
        shards = get_downloaded_shards()
        count = len(shards)
        size_gb = get_downloaded_size_gb()

        if count != last_count or int(time.time() - t0) % 60 == 0:
            print(f"[DOWNLOAD PROGRESS] {count}/{EXPECTED_SHARDS} shards ({size_gb:.2f} GB downloaded)...", flush=True)
            last_count = count

        if count >= EXPECTED_SHARDS:
            # Check if any .incomplete or lock files exist across raw dir and .cache
            incompletes = [os.path.join(r, f) for r, d, fs in os.walk(RAW_DIR) for f in fs if f.endswith(".incomplete")]
            if not incompletes and size_gb > 460:
                print(f"\n[DOWNLOAD COMPLETE] All {count} shards downloaded successfully ({size_gb:.2f} GB)!", flush=True)
                break

        time.sleep(15)

def run_conversion():
    print("\n" + "=" * 70)
    print("  [PIPELINE STEP 2] Packing Dense Backbone & 15,360 Experts (Direct I/O)")
    print("=" * 70, flush=True)

    cmd = [
        sys.executable,
        "scripts/convert_deepseek_v41.py",
        "--model-dir", str(RAW_DIR),
        "--out-dir", str(OUT_DIR)
    ]
    res = subprocess.run(cmd)
    if res.returncode != 0:
        print("[ERROR] Conversion failed!")
        sys.exit(1)
    print("[CONVERSION COMPLETE] Manifest and Direct I/O binaries ready.")

def test_inference():
    print("\n" + "=" * 70)
    print("  [PIPELINE STEP 3] Launching Moecher Server & Verifying Inference")
    print("=" * 70, flush=True)

    exe_path = Path(__file__).resolve().parent.parent / "moecher.exe"
    manifest_file = OUT_DIR / "moecher_manifest.json"
    server_cmd = [
        str(exe_path),
        "-m", str(manifest_file),
        "--cache-gb", "46",
        "--dram-cache-gb", "32",
        "--port", str(SERVER_PORT)
    ]

    print(f"Launching server: {' '.join(server_cmd)}")
    proc = subprocess.Popen(server_cmd)

    # Health check loop
    url = f"http://127.0.0.1:{SERVER_PORT}/v1/models"
    print("Waiting for server startup...")
    up = False
    for _ in range(90):
        time.sleep(2)
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    up = True
                    break
        except Exception:
            pass

    if not up:
        print("[ERROR] Server failed to start within timeout.")
        proc.kill()
        sys.exit(1)

    print("\n[SERVER ONLINE] Performing inference test...")
    chat_url = f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions"
    payload = json.dumps({
        "model": "deepseek-v4.1-flash",
        "messages": [
            {"role": "user", "content": "Explain the difference between Multi-Head Attention (MHA) and Multi-Head Latent Attention (MLA)."}
        ],
        "max_tokens": 128,
        "temperature": 0.0
    }).encode("utf-8")

    req = urllib.request.Request(chat_url, data=payload, headers={"Content-Type": "application/json"})
    t_start = time.time()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            elapsed = time.time() - t_start
            reply = data["choices"][0]["message"]["content"]
            tokens = data.get("usage", {}).get("completion_tokens", 0)
            tok_per_sec = tokens / elapsed if elapsed > 0 else 0

            print("\n" + "=" * 70)
            print("  [SMOKE TEST RESULT]")
            print(f"  Tokens Generated: {tokens}")
            print(f"  Time Elapsed:     {elapsed:.2f} s")
            print(f"  Throughput:       {tok_per_sec:.2f} tok/s")
            print("=" * 70)
            print(f"Response:\n{reply}\n")
    except Exception as e:
        print(f"[ERROR] Inference test error: {e}")
        proc.kill()
        sys.exit(1)

    print("[SUCCESS] DeepSeek-V4.1-Flash fully operational on RTX 6000 Pro 96GB!")

def main():
    wait_for_download()
    run_conversion()
    test_inference()

if __name__ == "__main__":
    main()
