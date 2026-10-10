import os
import sys
import time
import json
import base64
import urllib.request
import urllib.error
import subprocess
import re

TEST_IMAGE_PATH = r"e:\dev\MinnieTheMoEcher\graphics\moecher_logo_b_clear.jpg"
LOG_PATH = r"e:\dev\MinnieTheMoEcher\moecher.log"
MANIFEST_PATH = r"E:\moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json"
GGUF_MODEL_PATH = r"E:\moecher\models\gguf_qwen3_8_27b\Qwen3.8-27B-Q3_K_M.gguf"
GGUF_MMPROJ_PATH = r"E:\moecher\models\gguf_qwen3_8_27b\mmproj-Qwen3.8-27B-bf16.gguf"

def read_last_log_lines(n=50):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        return f.readlines()[-n:]

def extract_moecher_stats(log_lines):
    prefill_stat = None
    gen_stat = None
    spec_stat = None
    for line in reversed(log_lines):
        if not prefill_stat and "[PREFILL STATS]" in line:
            # [PREFILL STATS] 512 tokens evaluated in 0.812s -> 630.54 tok/s ...
            m = re.search(r"(\d+) tokens evaluated in ([\d\.]+)s -> ([\d\.]+) tok/s", line)
            if m:
                prefill_stat = {
                    "tokens": int(m.group(1)),
                    "sec": float(m.group(2)),
                    "tps": float(m.group(3))
                }
        if not gen_stat and "[GENERATION STATS]" in line:
            # [GENERATION STATS] 128 tokens in 4.321s -> 29.62 tok/s ...
            m = re.search(r"(\d+) tokens in ([\d\.]+)s -> ([\d\.]+) tok/s", line)
            if m:
                gen_stat = {
                    "tokens": int(m.group(1)),
                    "sec": float(m.group(2)),
                    "tps": float(m.group(3))
                }
        if not spec_stat and "[SPECULATIVE STATS]" in line:
            # [SPECULATIVE STATS] 85 cycles | 85 drafted | 39 accepted (45.9% acceptance rate) | draft: 177.0ms total (2.08ms/c) | verify: 4266.0ms total (50.19ms/c)
            m = re.search(r"(\d+) cycles \| (\d+) drafted \| (\d+) accepted \(([\d\.]+)%\) \| draft: ([\d\.]+)ms total \(([\d\.]+)ms/c\) \| verify: ([\d\.]+)ms total \(([\d\.]+)ms/c\)", line)
            if m:
                spec_stat = {
                    "cycles": int(m.group(1)),
                    "drafted": int(m.group(2)),
                    "accepted": int(m.group(3)),
                    "acceptance_rate": float(m.group(4)),
                    "draft_ms_per_c": float(m.group(6)),
                    "verify_ms_per_c": float(m.group(8))
                }
        if prefill_stat and gen_stat:
            break
    return prefill_stat, gen_stat, spec_stat

def get_gpu_memory(proc_name):
    cmd = [
        "powershell", "-ExecutionPolicy", "Bypass",
        "-File", r"e:\dev\MinnieTheMoEcher\scripts\measure_gpu.ps1",
        "-ProcessName", proc_name
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")
    for l in lines:
        parts = l.split()
        if len(parts) >= 5 and parts[0].lower().startswith(proc_name.lower()):
            try:
                ws = float(parts[2])
                ded = float(parts[3])
                sha = float(parts[4])
                return {"working_set_mb": ws, "dedicated_vram_mb": ded, "shared_pcie_mb": sha}
            except ValueError:
                pass
    return {"working_set_mb": 0, "dedicated_vram_mb": 0, "shared_pcie_mb": 0}

def wait_for_endpoint(url, timeout=90):
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            time.sleep(1.0)
    return False

def make_request(port, messages, max_tokens=128, temperature=0.0):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=180) as resp:
        t1 = time.time()
        res_body = json.loads(resp.read().decode("utf-8"))
        elapsed = t1 - t0
        usage = res_body.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        content = res_body.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "elapsed_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "content": content[:120].strip()
        }

def generate_text_of_length(words_target):
    base_snippet = "The development of hybrid transformer architectures combining delta linear attention and sparse state space models offers significant latency advantages on consumer hardware. "
    repeats = (words_target // len(base_snippet.split())) + 1
    return (base_snippet * repeats)[:words_target * 7]

print("Benchmark helper loaded successfully.")
