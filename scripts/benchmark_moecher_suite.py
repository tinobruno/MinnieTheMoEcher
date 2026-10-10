import os
import sys
import time
import json
import base64
import urllib.request
import subprocess
import re

TEST_IMAGE_PATH = r"e:\dev\MinnieTheMoEcher\graphics\moecher_logo_b_clear.jpg"
LOG_PATH = r"e:\dev\MinnieTheMoEcher\moecher.log"
MANIFEST_PATH = r"E:\moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json"

def get_gpu_memory(proc_name="moecher"):
    cmd = [
        "powershell", "-ExecutionPolicy", "Bypass",
        "-File", r"e:\dev\MinnieTheMoEcher\scripts\measure_gpu.ps1",
        "-ProcessName", proc_name
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")
    for l in lines:
        parts = l.split()
        if len(parts) >= 5 and proc_name.lower() in parts[0].lower():
            try:
                ws = float(parts[2])
                ded = float(parts[3])
                sha = float(parts[4])
                return {"working_set_mb": ws, "dedicated_vram_mb": ded, "shared_pcie_mb": sha}
            except ValueError:
                pass
    return {"working_set_mb": 0, "dedicated_vram_mb": 0, "shared_pcie_mb": 0}

def wait_for_server(url="http://127.0.0.1:8001/api/model", timeout=120):
    print(f"Waiting for server at {url}...")
    start = time.time()
    while time.time() - start < timeout:
        try:
            req = urllib.request.Request(url)
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    print(f"Server ready after {time.time() - start:.1f}s")
                    return True
        except Exception:
            time.sleep(1.0)
    print("Server wait timed out!")
    return False

def extract_latest_stats():
    if not os.path.exists(LOG_PATH):
        return None, None, None
    with open(LOG_PATH, "r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()[-40:]
    
    prefill = None
    gen = None
    spec = None
    for line in reversed(lines):
        if not prefill and "[PREFILL STATS]" in line:
            m = re.search(r"(\d+) tokens evaluated in ([\d\.]+)s -> ([\d\.]+) tok/s", line)
            if m:
                prefill = {"tokens": int(m.group(1)), "sec": float(m.group(2)), "tps": float(m.group(3))}
        if not gen and "[GENERATION STATS]" in line:
            m = re.search(r"(\d+) tokens in ([\d\.]+)s -> ([\d\.]+) tok/s", line)
            if m:
                gen = {"tokens": int(m.group(1)), "sec": float(m.group(2)), "tps": float(m.group(3))}
        if not spec and "[SPECULATIVE STATS]" in line:
            m = re.search(r"(\d+) cycles \| (\d+) drafted \| (\d+) accepted \(([\d\.]+)%\)", line)
            if m:
                spec = {"cycles": int(m.group(1)), "drafted": int(m.group(2)), "accepted": int(m.group(3)), "rate": float(m.group(4))}
    return prefill, gen, spec

def run_query(messages, max_tokens=128, temperature=0.0):
    url = "http://127.0.0.1:8001/v1/chat/completions"
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
    with urllib.request.urlopen(req, timeout=300) as resp:
        t1 = time.time()
        body = json.loads(resp.read().decode("utf-8"))
        elapsed = t1 - t0
        prefill, gen, spec = extract_latest_stats()
        mem = get_gpu_memory("moecher")
        return {
            "wall_sec": elapsed,
            "prefill": prefill,
            "generation": gen,
            "speculative": spec,
            "gpu_memory": mem,
            "response_preview": body["choices"][0]["message"]["content"][:100].strip()
        }

if __name__ == "__main__":
    print("Moecher benchmark client test ready.")
