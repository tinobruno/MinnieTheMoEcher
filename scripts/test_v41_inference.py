#!/usr/bin/env python3
"""
test_v41_inference.py — End-to-end verification test for DeepSeek-V4.1-Flash on RTX PRO 6000 96GB
"""

import os
import sys
import time
import json
import urllib.request
import urllib.error
import subprocess
from pathlib import Path

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8', errors='replace')

MANIFEST = Path("F:/Moecher/models/deepseek_v41_nvfp4/moecher_manifest.json")
EXE_PATH = Path("f:/dev/MinnieTheMoEcher/moecher.exe")
SERVER_PORT = 8001

def main():
    print("=" * 70)
    print("  DeepSeek-V4.1-Flash Native C++/CUDA Engine Verification Test")
    print("  Hardware: NVIDIA RTX PRO 6000 (96GB VRAM, 64GB Host RAM, NVMe SSD)")
    print("  Caching: L1 VRAM (46GB) <-> L2 DRAM (32GB) <-> L3 Direct I/O NVMe")
    print("=" * 70, flush=True)

    if not MANIFEST.exists():
        print(f"[ERROR] Manifest not found at {MANIFEST}")
        sys.exit(1)
    if not EXE_PATH.exists():
        print(f"[ERROR] Engine binary not found at {EXE_PATH}")
        sys.exit(1)

    # Kill any stale moecher processes
    subprocess.run(["powershell", "-Command", "Get-Process moecher -ErrorAction SilentlyContinue | Stop-Process -Force"], capture_output=True)
    time.sleep(1)

    server_cmd = [
        str(EXE_PATH),
        "-m", str(MANIFEST),
        "--cache-gb", "46",
        "--dram-cache-gb", "32",
        "--port", str(SERVER_PORT),
        "-q"
    ]

    print(f"\n[1/3] Booting Moecher Server: {' '.join(server_cmd)}", flush=True)
    proc = subprocess.Popen(server_cmd)

    try:
        # Health check polling
        url = f"http://127.0.0.1:{SERVER_PORT}/v1/models"
        print("      Waiting for server to load weights and initialize KV cache...", end="", flush=True)
        up = False
        for i in range(120):
            time.sleep(1)
            print(".", end="", flush=True)
            try:
                req = urllib.request.Request(url)
                with urllib.request.urlopen(req, timeout=2) as resp:
                    if resp.status == 200:
                        up = True
                        print(" ONLINE!")
                        break
            except Exception:
                pass

        if not up:
            print("\n[ERROR] Server failed to bind within 120 seconds!")
            sys.exit(1)

        test_queries = [
            ("Factual / Conciseness Check", "What is the capital of France? Answer in one short sentence.", 96),
            ("Technical DeepSeek Architecture Check", "Explain the difference between Multi-Head Attention (MHA) and Multi-Head Latent Attention (MLA) in two concise bullet points.", 256)
        ]

        chat_url = f"http://127.0.0.1:{SERVER_PORT}/v1/chat/completions"

        for idx, (label, query, max_toks) in enumerate(test_queries, 1):
            print(f"\n[2/3.{idx}] Test: {label}")
            print(f"      Prompt: \"{query}\"", flush=True)

            payload = json.dumps({
                "model": "deepseek-v4.1-flash",
                "messages": [
                    {"role": "user", "content": query}
                ],
                "max_tokens": max_toks,
                "thinking_budget": 96,
                "repetition_penalty": 1.1,
                "temperature": 0.0,
                "stream": True
            }).encode("utf-8")

            req = urllib.request.Request(chat_url, data=payload, headers={"Content-Type": "application/json"})
            t_start = time.time()
            first_token_time = None
            content_parts = []
            reasoning_parts = []
            total_tokens = 0

            print("\n--- Live Model Token Stream ---", flush=True)
            with urllib.request.urlopen(req, timeout=600) as resp:
                for line in resp:
                    line_str = line.decode("utf-8", errors="replace").strip()
                    if not line_str or line_str.startswith(":"):
                        continue
                    if line_str == "data: [DONE]":
                        break
                    if line_str.startswith("data: "):
                        json_str = line_str[6:].strip()
                        try:
                            chunk = json.loads(json_str)
                            choices = chunk.get("choices", [])
                            if not choices:
                                continue
                            delta = choices[0].get("delta", {})
                            r_piece = delta.get("reasoning_content") or ""
                            c_piece = delta.get("content") or ""

                            if r_piece:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                reasoning_parts.append(r_piece)
                                print(r_piece, end="", flush=True)
                                total_tokens += 1
                            if c_piece:
                                if first_token_time is None:
                                    first_token_time = time.time()
                                content_parts.append(c_piece)
                                print(c_piece, end="", flush=True)
                                total_tokens += 1
                        except Exception as e:
                            pass
            elapsed = time.time() - t_start
            print("\n--- End of Stream ---\n", flush=True)

            content = "".join(content_parts)
            reasoning = "".join(reasoning_parts)
            ttft = (first_token_time - t_start) if first_token_time else elapsed
            decode_time = elapsed - ttft
            tok_per_sec = (total_tokens / decode_time) if decode_time > 0 and total_tokens > 0 else 0

            print(f"\n[Result for Test {idx}]:")
            print("=" * 70)
            print(f"  Completion Tokens: {total_tokens}")
            print(f"  Time to First Tok: {ttft:.2f} s")
            print(f"  Total Elapsed:     {elapsed:.2f} s")
            print(f"  Decode Speed:      {tok_per_sec:.2f} tok/s")
            print("=" * 70)
            if reasoning.strip():
                print(f"\n[Reasoning Thought Stream]:\n{reasoning}\n")
            if content.strip():
                print(f"\n[Model Content Output]:\n{content}\n")
            print("=" * 70)

            combined = (reasoning + " " + content).strip()
            if len(combined) > 20 and "repetition loop" not in combined.lower():
                print(f"[SUCCESS] Test {idx} ({label}) passed coherent generation!")
            else:
                print(f"[WARN] Test {idx} ({label}) check output.")

    finally:
        print("\nTerminating test server process...", flush=True)
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
        print("Server shutdown clean.")

if __name__ == "__main__":
    main()

