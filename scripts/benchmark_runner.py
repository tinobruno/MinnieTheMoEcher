import sys
import time
import json
import base64
import urllib.request
import urllib.error
import subprocess

def encode_image_base64(path):
    with open(path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

def wait_for_server(url, timeout=90):
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

def get_gpu_memory(proc_name):
    cmd = [
        "powershell", "-ExecutionPolicy", "Bypass",
        "-File", "e:\\dev\\MinnieTheMoEcher\\scripts\\measure_gpu.ps1",
        "-ProcessName", proc_name
    ]
    res = subprocess.run(cmd, capture_output=True, text=True)
    lines = res.stdout.strip().split("\n")
    # Parse table output
    for l in lines:
        parts = l.split()
        if len(parts) >= 5 and parts[0] == proc_name:
            try:
                ws = float(parts[2])
                ded = float(parts[3])
                sha = float(parts[4])
                return {"working_set_mb": ws, "dedicated_vram_mb": ded, "shared_pcie_mb": sha}
            except ValueError:
                pass
    return {"working_set_mb": 0, "dedicated_vram_mb": 0, "shared_pcie_mb": 0}

def send_chat_completion(port, messages, max_tokens=128, temperature=0.0):
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
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            t1 = time.time()
            res_body = json.loads(resp.read().decode("utf-8"))
            elapsed = t1 - t0
            usage = res_body.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            content = res_body.get("choices", [{}])[0].get("message", {}).get("content", "")
            return {
                "success": True,
                "elapsed_sec": elapsed,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": prompt_tokens + completion_tokens,
                "content_preview": content[:120].strip(),
                "response_json": res_body
            }
    except Exception as e:
        return {"success": False, "error": str(e)}

if __name__ == "__main__":
    print("Benchmark runner module ready.")
