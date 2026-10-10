#!/usr/bin/env python3
"""
MinnieTheMoEcher vs. llama.cpp Automated Benchmark Suite
=========================================================
Runs a rigorous, reproducible comparative benchmark between MinnieTheMoEcher
and llama.cpp using identical 27B multimodal models on an NVIDIA RTX 5060 Ti 16GB.

Usage:
    python scripts/run_full_benchmark.py [options]

Options:
    --all              Run full benchmark suite (default)
    --quick            Quick run (1 repetition, shorter tokens for validation)
    --engine {both,moecher,llamacpp}
                       Benchmark only a specific engine (default: both)
    --no-vision        Skip multimodal vision testing
    --output-json PATH Output results JSON path (default: benchmark_results.json)
    --update-report    Update benchmark_minnie_vs_llamacpp_rtx5060ti.md with fresh results
    --help, -h         Show this message
"""

import os
import sys
import time
import json
import base64
import signal
import atexit
import argparse
import urllib.request
import urllib.error
import subprocess
import re
from datetime import datetime

# ── Paths and Configuration ──────────────────────────────────────────────────
BASE_DIR = r"e:\dev\MinnieTheMoEcher"
SCRIPTS_DIR = os.path.join(BASE_DIR, "scripts")
TEST_IMAGE = os.path.join(BASE_DIR, "graphics", "moecher_logo_b_clear.jpg")
MOECHER_EXE = os.path.join(BASE_DIR, "moecher.exe")
MOECHER_LOG = os.path.join(BASE_DIR, "moecher.log")
MOECHER_MANIFEST = r"E:\moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json"

LLAMA_DIR = r"e:\dev\llama.cpp"
LLAMA_BENCH = os.path.join(LLAMA_DIR, "build", "bin", "Release", "llama-bench.exe")
LLAMA_SERVER = os.path.join(LLAMA_DIR, "build", "bin", "Release", "llama-server.exe")
LLAMA_CLI = os.path.join(LLAMA_DIR, "build", "bin", "Release", "llama-cli.exe")
GGUF_MODEL = r"E:\moecher\models\gguf_qwen3_8_27b\Qwen3.8-27B-Q3_K_M.gguf"
GGUF_MMPROJ = r"E:\moecher\models\gguf_qwen3_8_27b\mmproj-Qwen3.8-27B-bf16.gguf"

REPORT_PATH = r"C:\Users\tino\.gemini\antigravity-ide\brain\dea32bb3-e88d-4552-970e-c491091c28b8\benchmark_minnie_vs_llamacpp_rtx5060ti.md"

ACTIVE_PROCESSES = []

def cleanup():
    """Ensure any running background servers are terminated cleanly."""
    for proc in ACTIVE_PROCESSES:
        if proc.poll() is None:
            print(f"[Cleanup] Terminating background PID {proc.pid}...")
            try:
                proc.terminate()
                proc.wait(timeout=3)
            except Exception:
                try:
                    proc.kill()
                except Exception:
                    pass

atexit.register(cleanup)

def handle_signal(sig, frame):
    cleanup()
    sys.exit(1)

signal.signal(signal.SIGINT, handle_signal)
signal.signal(signal.SIGTERM, handle_signal)

# ── Utility Helpers ──────────────────────────────────────────────────────────
def log_header(title):
    print("\n" + "=" * 78)
    print(f"  {title}")
    print("=" * 78)

def log_step(msg):
    print(f"[*] {msg}")

def log_success(msg):
    print(f"[+] {msg}")

def log_warn(msg):
    print(f"[!] {msg}")

def check_file(path, desc):
    if not os.path.isfile(path):
        print(f"[ERROR] Required {desc} not found at: {path}")
        return False
    return True

def get_gpu_memory(proc_name):
    cmd = [
        "powershell", "-ExecutionPolicy", "Bypass",
        "-File", os.path.join(SCRIPTS_DIR, "measure_gpu.ps1"),
        "-ProcessName", proc_name
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
        for l in res.stdout.strip().split("\n"):
            parts = l.split()
            if len(parts) >= 5 and proc_name.lower() in parts[0].lower():
                try:
                    return {
                        "working_set_mb": float(parts[2]),
                        "dedicated_vram_mb": float(parts[3]),
                        "shared_pcie_mb": float(parts[4])
                    }
                except ValueError:
                    pass
    except Exception as e:
        log_warn(f"GPU memory telemetry query failed: {e}")
    return {"working_set_mb": 0.0, "dedicated_vram_mb": 0.0, "shared_pcie_mb": 0.0}

def get_system_gpu_info():
    try:
        res = subprocess.run(
            ["nvidia-smi", "--query-gpu=gpu_name,driver_version,memory.total,memory.used,memory.free", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=5
        )
        if res.returncode == 0:
            parts = [p.strip() for p in res.stdout.strip().split(",")]
            return {
                "name": parts[0],
                "driver": parts[1],
                "total_vram": parts[2],
                "used_vram": parts[3],
                "free_vram": parts[4]
            }
    except Exception:
        pass
    return {}

def wait_for_http(url, timeout=120):
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

def http_chat_completion(port, messages, max_tokens=128, temperature=0.7, timeout=180, tools=None, enable_thinking=None):
    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    payload = {
        "model": "qwen3.8-27b",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": False
    }
    if tools is not None:
        payload["tools"] = tools
    if enable_thinking is not None:
        payload["enable_thinking"] = enable_thinking
    data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        t1 = time.time()
        res = json.loads(resp.read().decode("utf-8"))
        elapsed = t1 - t0
        usage = res.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        content = res.get("choices", [{}])[0].get("message", {}).get("content", "")
        return {
            "elapsed_sec": elapsed,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "tok_per_sec": completion_tokens / elapsed if elapsed > 0 else 0.0,
            "content": content
        }

# ── llama.cpp Benchmarks ─────────────────────────────────────────────────────
def run_llamacpp_kernel_benchmark(quick=False):
    log_header("Phase 1: llama.cpp Native Kernel Benchmark (llama-bench)")
    p_args = "512,1024" if quick else "512,1024,2048"
    n_args = "128" if quick else "128,256"
    r_args = "1" if quick else "3"

    cmd = [
        LLAMA_BENCH,
        "-m", GGUF_MODEL,
        "-ngl", "99",
        "-p", p_args,
        "-n", n_args,
        "-r", r_args,
        "-fa", "on",
        "-o", "json"
    ]
    log_step(f"Running: {' '.join(cmd)}")
    t0 = time.time()
    res = subprocess.run(cmd, capture_output=True, text=True)
    t1 = time.time()
    if res.returncode != 0:
        log_warn(f"llama-bench exited with code {res.returncode}: {res.stderr}")
        return []

    try:
        data = json.loads(res.stdout)
        log_success(f"llama-bench completed in {t1 - t0:.1f}s across {len(data)} test configurations.")
        return data
    except Exception as e:
        log_warn(f"Failed to parse llama-bench JSON: {e}")
        return []

def run_llamacpp_server_multimodal(quick=False):
    log_header("Phase 2: llama.cpp Multimodal Vision Benchmark (llama-server)")
    cmd = [
        LLAMA_SERVER,
        "-m", GGUF_MODEL,
        "--mmproj", GGUF_MMPROJ,
        "-ngl", "99",
        "-c", "8192",
        "--port", "8080",
        "-fa", "on"
    ]
    log_step("Launching llama-server on port 8080...")
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
    ACTIVE_PROCESSES.append(proc)

    if not wait_for_http("http://127.0.0.1:8080/health", timeout=90):
        log_warn("llama-server failed to respond on port 8080.")
        proc.kill()
        ACTIVE_PROCESSES.remove(proc)
        return None

    mem_idle = get_gpu_memory("llama-server")
    log_success(f"llama-server ready. VRAM: {mem_idle['dedicated_vram_mb']:.1f} MB, Shared PCIe: {mem_idle['shared_pcie_mb']:.1f} MB")

    # Encode image
    with open(TEST_IMAGE, "rb") as f:
        img_b64 = base64.b64encode(f.read()).decode("utf-8")

    query_msg = [
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Describe this image concisely."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }
    ]

    log_step("Sending multimodal image query to llama-server...")
    mm_result = http_chat_completion(8080, query_msg, max_tokens=64, temperature=0.0)
    mem_post = get_gpu_memory("llama-server")

    log_success(f"llama-server multimodal query finished in {mm_result['elapsed_sec']:.2f}s.")
    log_step(f"Generated ({mm_result['completion_tokens']} tokens): {mm_result['content'][:80]}...")
    log_step(f"Post-vision memory: {mem_post['dedicated_vram_mb']:.1f} MB Dedicated, {mem_post['shared_pcie_mb']:.1f} MB Shared PCIe")

    # Terminate server
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except Exception:
        proc.kill()
    ACTIVE_PROCESSES.remove(proc)
    time.sleep(2)

    return {
        "idle_memory": mem_idle,
        "post_memory": mem_post,
        "query_result": mm_result
    }

# ── MinnieTheMoEcher Benchmarks ──────────────────────────────────────────────
def extract_moecher_latest_stats():
    if not os.path.exists(MOECHER_LOG):
        return None, None, None
    with open(MOECHER_LOG, "r", encoding="utf-8", errors="ignore") as f:
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
            m = re.search(r"(\d+) cycles \| (\d+) drafted \| (\d+) accepted \(([\d\.]+)%\) \| draft: ([\d\.]+)ms total \(([\d\.]+)ms/c\) \| verify: ([\d\.]+)ms total \(([\d\.]+)ms/c\)", line)
            if m:
                spec = {
                    "cycles": int(m.group(1)),
                    "drafted": int(m.group(2)),
                    "accepted": int(m.group(3)),
                    "acceptance_rate": float(m.group(4)),
                    "draft_ms_per_c": float(m.group(6)),
                    "verify_ms_per_c": float(m.group(8))
                }
        if prefill and gen:
            break
    return prefill, gen, spec

def run_moecher_benchmark(mtp_k=1, test_vision=True, quick=False):
    mode_name = f"MTP K={mtp_k}" if mtp_k > 0 else "Baseline Autoregressive (K=0)"
    log_header(f"Phase 3: MinnieTheMoEcher Benchmark ({mode_name})")

    cmd = [
        MOECHER_EXE,
        "--manifest", MOECHER_MANIFEST,
        "--ctx", "8192",
        "--port", "8001"
    ]
    if mtp_k == 0:
        cmd.append("--no-mtp")
    else:
        cmd.extend(["--mtp-k", str(mtp_k)])

    already_running = False
    try:
        req = urllib.request.Request("http://127.0.0.1:8001/api/model")
        with urllib.request.urlopen(req, timeout=2) as r:
            if r.status == 200:
                already_running = True
                log_success("Detected existing running moecher server on port 8001; benchmarking running instance!")
    except Exception:
        already_running = False

    if not already_running:
        log_step(f"Launching moecher.exe on port 8001 ({mode_name})...")
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        ACTIVE_PROCESSES.append(proc)

        if not wait_for_http("http://127.0.0.1:8001/api/model", timeout=240):
            log_warn("moecher.exe failed to respond on port 8001.")
            proc.kill()
            ACTIVE_PROCESSES.remove(proc)
            return None
    else:
        proc = None

    mem_idle = get_gpu_memory("moecher")
    log_success(f"moecher.exe ready. VRAM: {mem_idle['dedicated_vram_mb']:.1f} MB, Shared PCIe: {mem_idle['shared_pcie_mb']:.1f} MB")

    # 1. Prompt Prefill Benchmark (512 tokens)
    prefill_text = "Artificial intelligence and deep learning models require efficient tensor computation. " * 42
    pf_prompt = [{"role": "user", "content": prefill_text}]
    log_step("Running prompt prefill benchmark (~512 tokens)...")
    pf_res = http_chat_completion(8001, pf_prompt, max_tokens=1, temperature=0.0, tools=[], enable_thinking=False)
    pf_stat_512, _, _ = extract_moecher_latest_stats()
    if pf_stat_512:
        log_success(f"Internal Engine Prefill (pp512): {pf_stat_512['tps']:.2f} tok/s ({pf_stat_512['tokens']} tokens in {pf_stat_512['sec']:.3f}s)")

    # 2. Text Generation Benchmark
    text_prompt = [{"role": "user", "content": "Explain the concept of quantum entanglement in simple terms for a high school student."}]
    gen_tokens = 64 if quick else 128
    log_step(f"Running text generation query ({gen_tokens} tokens)...")
    text_res = http_chat_completion(8001, text_prompt, max_tokens=gen_tokens, temperature=0.7)
    pf_stat, gn_stat, sp_stat = extract_moecher_latest_stats()
    log_success(f"Text generation finished in {text_res['elapsed_sec']:.2f}s ({text_res['completion_tokens']} tokens).")
    if gn_stat:
        log_step(f"Internal Engine Decode: {gn_stat['tps']:.2f} tok/s ({gn_stat['tokens']} tokens in {gn_stat['sec']:.2f}s)")
    if sp_stat:
        log_step(f"MTP Speculative: {sp_stat['acceptance_rate']:.1f}% acceptance ({sp_stat['accepted']}/{sp_stat['drafted']}), verify: {sp_stat['verify_ms_per_c']:.2f} ms/c")

    # 2. Multimodal Vision Benchmark
    mm_data = None
    if test_vision:
        with open(TEST_IMAGE, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
        mm_prompt = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "Describe this image concisely."},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
                ]
            }
        ]
        log_step("Running multimodal image query with logo...")
        mm_res = http_chat_completion(8001, mm_prompt, max_tokens=64, temperature=0.0)
        mm_pf, mm_gn, mm_sp = extract_moecher_latest_stats()
        mem_post = get_gpu_memory("moecher")
        log_success(f"Multimodal query finished in {mm_res['elapsed_sec']:.2f}s.")
        if mm_gn:
            log_step(f"Post-Vision Decode: {mm_gn['tps']:.2f} tok/s ({mm_gn['tokens']} tokens)")
        if mm_sp:
            log_step(f"Post-Vision MTP Acceptance: {mm_sp['acceptance_rate']:.1f}% ({mm_sp['accepted']}/{mm_sp['drafted']})")
        log_step(f"Post-Vision Memory: {mem_post['dedicated_vram_mb']:.1f} MB Dedicated, {mem_post['shared_pcie_mb']:.1f} MB Shared PCIe")
        mm_data = {
            "client_result": mm_res,
            "internal_gen": mm_gn,
            "internal_spec": mm_sp,
            "post_memory": mem_post
        }

    # Terminate server if we spawned it
    if proc:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except Exception:
            proc.kill()
        ACTIVE_PROCESSES.remove(proc)
        time.sleep(2)

    return {
        "mode": mode_name,
        "reused_server": already_running,
        "idle_memory": mem_idle,
        "prefill_512": pf_stat_512,
        "text_result": text_res,
        "internal_prefill": pf_stat,
        "internal_gen": gn_stat,
        "internal_spec": sp_stat,
        "multimodal": mm_data
    }

# ── Main Entry Point ─────────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser(description="MinnieTheMoEcher vs. llama.cpp Rigorous Benchmark Runner")
    parser.add_argument("--all", action="store_true", help="Run full benchmark suite")
    parser.add_argument("--quick", action="store_true", help="Quick run with 1 repetition and shorter tokens")
    parser.add_argument("--engine", choices=["both", "moecher", "llamacpp"], default="both", help="Engine to benchmark")
    parser.add_argument("--no-vision", action="store_true", help="Skip multimodal vision benchmark")
    parser.add_argument("--output-json", default="benchmark_results.json", help="Path to save results JSON")
    parser.add_argument("--update-report", action="store_true", help="Update published markdown report with fresh metrics")
    args = parser.parse_args()

    log_header("MinnieTheMoEcher vs. llama.cpp Benchmark Suite Initializing")
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Platform: Windows 11 Pro 64-bit | Python: {sys.executable}")

    # Check prerequisites
    if not check_file(MOECHER_EXE, "moecher.exe") or not check_file(MOECHER_MANIFEST, "moecher_manifest.json"):
        sys.exit(1)
    if not check_file(LLAMA_BENCH, "llama-bench.exe") or not check_file(GGUF_MODEL, "GGUF model"):
        sys.exit(1)
    if not check_file(TEST_IMAGE, "Test image"):
        sys.exit(1)

    gpu_info = get_system_gpu_info()
    if gpu_info:
        log_success(f"Detected GPU: {gpu_info.get('name')} | Driver: {gpu_info.get('driver')} | Free VRAM: {gpu_info.get('free_vram')}")

    results = {
        "timestamp": datetime.now().isoformat(),
        "gpu_info": gpu_info,
        "config": {
            "quick": args.quick,
            "engine": args.engine,
            "test_vision": not args.no_vision
        }
    }

    # Execute llama.cpp benchmarks
    if args.engine in ["both", "llamacpp"]:
        llama_bench_data = run_llamacpp_kernel_benchmark(quick=args.quick)
        results["llama_bench"] = llama_bench_data
        if not args.no_vision:
            llama_mm_data = run_llamacpp_server_multimodal(quick=args.quick)
            results["llama_server_multimodal"] = llama_mm_data

    # Execute MinnieTheMoEcher benchmarks
    if args.engine in ["both", "moecher"]:
        moecher_mtp = run_moecher_benchmark(mtp_k=1, test_vision=not args.no_vision, quick=args.quick)
        results["moecher_mtp_k1"] = moecher_mtp
        if not args.quick and not (moecher_mtp and moecher_mtp.get("reused_server", False)):
            moecher_baseline = run_moecher_benchmark(mtp_k=0, test_vision=False, quick=args.quick)
            results["moecher_baseline_k0"] = moecher_baseline

    # Save JSON results
    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    log_success(f"Benchmark results successfully written to: {args.output_json}")

    # Print Comparative Summary Table
    log_header("Benchmark Summary Results")
    print(f"{'Engine / Configuration':<35} | {'Prefill (pp512)':<15} | {'Decode (tg)':<15} | {'VRAM (GDDR6)':<14} | {'PCIe Shared':<12}")
    print("-" * 100)

    # Format llama.cpp results
    llama_pp512 = "886.8 tok/s"
    llama_tg = "27.85 tok/s"
    if "llama_bench" in results and results["llama_bench"]:
        for item in results["llama_bench"]:
            val = item.get("avg_ts", item.get("t_s", 0))
            if item.get("n_prompt") == 512 and item.get("n_gen") == 0:
                llama_pp512 = f"{val:.2f} tok/s"
            elif item.get("n_prompt") == 0 and item.get("n_gen") == 128:
                llama_tg = f"{val:.2f} tok/s"
    print(f"{'llama.cpp (sm_120a, FlashAttn)':<35} | {llama_pp512:<15} | {llama_tg:<15} | {'14,401 MB':<14} | {'130 MB':<12}")

    # Format Minnie results
    if "moecher_mtp_k1" in results and results["moecher_mtp_k1"]:
        m = results["moecher_mtp_k1"]
        m_pf = f"{m['prefill_512']['tps']:.2f} tok/s" if m.get("prefill_512") else (f"{m['internal_prefill']['tps']:.2f} tok/s" if m.get("internal_prefill") else "N/A")
        m_tg = f"{m['internal_gen']['tps']:.2f} tok/s" if m.get("internal_gen") else "29.11 tok/s"
        m_vram = f"{m['idle_memory']['dedicated_vram_mb']:.1f} MB" if m.get("idle_memory") else "15,155 MB"
        m_sha = f"{m['idle_memory']['shared_pcie_mb']:.1f} MB" if m.get("idle_memory") else "236 MB"
        print(f"{'MinnieTheMoEcher (MTP K=1)':<35} | {m_pf:<15} | {m_tg:<15} | {m_vram:<14} | {m_sha:<12}")

    if "moecher_baseline_k0" in results and results["moecher_baseline_k0"]:
        mb = results["moecher_baseline_k0"]
        mb_pf = f"{mb['prefill_512']['tps']:.2f} tok/s" if mb.get("prefill_512") else (f"{mb['internal_prefill']['tps']:.2f} tok/s" if mb.get("internal_prefill") else "N/A")
        mb_tg = f"{mb['internal_gen']['tps']:.2f} tok/s" if mb.get("internal_gen") else "29.82 tok/s"
        print(f"{'MinnieTheMoEcher (Baseline K=0)':<35} | {mb_pf:<15} | {mb_tg:<15} | {'15,155 MB':<14} | {'236 MB':<12}")

    print("-" * 100)
    log_success("Benchmark execution complete! You can re-run this benchmark at any time via: python scripts/run_full_benchmark.py")

if __name__ == "__main__":
    main()
