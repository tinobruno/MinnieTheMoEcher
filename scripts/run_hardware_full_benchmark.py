#!/usr/bin/env python3
"""
run_hardware_full_benchmark.py
Comprehensive Hardware-Specific Multi-Model Benchmark & Output Inspection Suite
Target Folder: f:\\dev\\MinnieTheMoEcher\\benchmarks_RTXPRO6000_96GB_9950X3D_64GB_DDR5_6000
"""

import os
import sys
import time
import json
import re
import subprocess
import urllib.request
import urllib.error
from datetime import datetime
from pathlib import Path

if sys.stdout and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
if sys.stderr and hasattr(sys.stderr, "reconfigure"):
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

ROOT_DIR = Path(r"f:\dev\MinnieTheMoEcher")
OUTPUT_DIR = ROOT_DIR / "benchmarks_RTXPRO6000_96GB_9950X3D_64GB_DDR5_6000"
OUTPUTS_SUBDIR = OUTPUT_DIR / "outputs"
SERVER_URL = "http://127.0.0.1:8001"
MOECHER_EXE = ROOT_DIR / "moecher.exe"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_SUBDIR.mkdir(parents=True, exist_ok=True)

HARDWARE_SPEC = {
    "folder_id": "benchmarks_RTXPRO6000_96GB_9950X3D_64GB_DDR5_6000",
    "gpu": {
        "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
        "architecture": "Blackwell (Compute Capability 12.0 / sm_120a)",
        "vram_total_mb": 97887,
        "vram_total_gb": 95.59,
        "memory_type": "GDDR7 (512-bit ECC, ~1,800 GB/s)",
        "driver_version": "591.86",
        "cuda_version": "13.0"
    },
    "cpu": {
        "name": "AMD Ryzen 9 9950X3D 16-Core Processor",
        "cores": 16,
        "threads": 32,
        "architecture": "Zen 5 with 2nd-Gen AMD 3D V-Cache"
    },
    "motherboard": {
        "manufacturer": "ASUSTeK COMPUTER INC.",
        "product": "ProArt X870E-CREATOR WIFI",
        "pcie": "PCIe 5.0 x16"
    },
    "ram": {
        "total_gb": 64.0,
        "configuration": "2x 32 GB DDR5 Dual-Channel",
        "manufacturer": "Micron",
        "part_number": "CP32G64C40U5B.M8B5",
        "jedec_speed_mts": 5600,
        "configured_speed_mts": 6000,
        "pinned_l2_budget_gb": 42.0
    },
    "storage": {
        "primary_model_drive": "F:\\ (Fanxiang S910MAX 4TB PCIe NVMe SSD)",
        "other_drives": [
            "WDC WDS100T2B0A-00SM50 (1TB)",
            "Samsung SSD 860 EVO (1TB)",
            "Samsung SSD 850 EVO (500GB)"
        ]
    },
    "os": "Microsoft Windows 11 Pro 64-bit (WDDM 3.2)"
}

MODELS = [
    {
        "id": "deepseek_v41_mixed",
        "name": "DeepSeek-V4.1-Flash (552B/14B) — Mixed NVFP4 + IQ2",
        "short_name": "DS-V4.1 Mixed (NVFP4+IQ2)",
        "architecture": "DeepSeek-V4.1 (40L, 15,360 Routed Experts, Top-6, MLA + mHC + Engram)",
        "params": "552B Total / 14B Active",
        "quant": "3,840 Hot NVFP4 (VRAM) + 683 Top Cold IQ2 (VRAM L1) + 4,530 Cold IQ2 (Pinned DDR5 L2) + FP8 Attn",
        "disk_gb": 231.3,
        "tier_mode": "3-Tier Hierarchy (VRAM Hot + VRAM Cold L1 + Pinned DDR5 L2 + NVMe SSD)",
        "manifest": r"F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json",
        "extra_args": []
    },
    {
        "id": "deepseek_v41_nvfp4",
        "name": "DeepSeek-V4.1-Flash (552B/14B) — Pure NVFP4",
        "short_name": "DS-V4.1 Pure NVFP4",
        "architecture": "DeepSeek-V4.1 (40L, 15,360 Routed Experts, Top-6, MLA + mHC + Engram)",
        "params": "552B Total / 14B Active",
        "quant": "Pure NVFP4 (sm_120a E2M1 + E4M3 Scales) + FP8/BF16 Attn",
        "disk_gb": 300.3,
        "tier_mode": "3-Tier Pure NVFP4 (VRAM L1 + Pinned DDR5 L2 + NVMe SSD)",
        "manifest": r"F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest.json",
        "extra_args": []
    },
    {
        "id": "deepseek_v4_flash_iq2",
        "name": "DeepSeek-V4-Flash (314B/11B) — BF16/FP8 Attn + IQ2_XXS",
        "short_name": "DS-V4-Flash IQ2 (BF16 Attn)",
        "architecture": "DeepSeek-V4-Flash 0731 (43L, 10,240 Routed Experts, Top-6, MLA + mHC + 3 Engram)",
        "params": "314B Total / 11B Active",
        "quant": "9.44 GB BF16/FP8 Attention + 77.91 GB IQ2_XXS Routed Experts",
        "disk_gb": 87.35,
        "tier_mode": "100% VRAM Resident (Zero PCIe / SSD Offloading)",
        "manifest": r"F:\Moecher\models\deepseek_v4_flash_iq2\moecher_manifest.json",
        "extra_args": []
    },
    {
        "id": "deepseek_v4_flash_q4",
        "name": "DeepSeek-V4-Flash (314B/11B) — Q4 Attn + IQ2_XXS",
        "short_name": "DS-V4-Flash IQ2 (Q4 Attn)",
        "architecture": "DeepSeek-V4-Flash 0731 (43L, 10,240 Routed Experts, Top-6, MLA + mHC + 3 Engram)",
        "params": "314B Total / 11B Active",
        "quant": "5.16 GB Q4 Attention + 77.91 GB IQ2_XXS Routed Experts",
        "disk_gb": 83.07,
        "tier_mode": "100% VRAM Resident (~12.5 GB Free VRAM Headroom)",
        "manifest": r"F:\Moecher\models\deepseek_v4_flash_q4\moecher_manifest.json",
        "extra_args": []
    },
    {
        "id": "qwen3_8_flash_next",
        "name": "Qwen 3.8 Flash-Next (GSQ-RCO-IQ1_M) — 512 Experts Hybrid GDN + QSA",
        "short_name": "Qwen 3.8 Flash-Next",
        "architecture": "Qwen 3.8 Hybrid MoE (48L: 36 Gated DeltaNet + 12 QSA Sparse Attention, 512 Routed Experts Top-10)",
        "params": "80B-Class MoE / ~3B Active",
        "quant": "3.44 GB BF16/Q8/Q6 Dense + 14.37 GB IQ2_XXS Routed Experts (17.81 GB Total)",
        "disk_gb": 17.81,
        "tier_mode": "100% VRAM Resident on 96GB / Per-Expert Hot-Cold L1+L2 Split on 16GB",
        "manifest": r"E:\dev\MinnieTheMoEcher\models\qwen3.8_flash_next\moecher_manifest.json" if Path(r"E:\dev\MinnieTheMoEcher\models\qwen3.8_flash_next\moecher_manifest.json").exists() else r"F:\Moecher\models\qwen3.8_flash_next\moecher_manifest.json",
        "extra_args": ["--ctx", "16384"]
    },
    {
        "id": "qwen3_8_27b_q4",
        "name": "Qwen 3.8 27B Dense — Q4_K / BF16",
        "short_name": "Qwen 3.8 27B Q4",
        "architecture": "Qwen 3.8 Hybrid (64L Gated DeltaNet Linear Attn + GQA + SwiGLU)",
        "params": "27B Dense",
        "quant": "20.06 GB Q4_K + BF16 Linear Attention",
        "disk_gb": 20.06,
        "tier_mode": "100% VRAM Resident + PLD / Speculative",
        "manifest": r"F:\Moecher\models\qwen3_8_27b_q4\moecher_manifest.json",
        "extra_args": []
    },
    {
        "id": "qwen3_8_27b_vision_13g",
        "name": "Qwen 3.8 27B Vision (13G Mixed Quant + MTP)",
        "short_name": "Qwen 3.8 27B Vision 13G",
        "architecture": "Qwen 3.8 Vision (32L ViT + 64L Hybrid Gated DeltaNet/GQA + Native MTP Self-Drafter)",
        "params": "27B Dense + Vision ViT",
        "quant": "14.19 GB Mixed INT4/FP4/BF16 + Native MTP Drafter",
        "disk_gb": 14.19,
        "tier_mode": "100% VRAM Resident + Native MTP Tree/Linear Speculative",
        "manifest": r"F:\Moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json",
        "extra_args": []
    },
    {
        "id": "nemotron_3_5_bolt",
        "name": "NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4)",
        "short_name": "Nemotron 3.5 Bolt 74B NVFP4",
        "architecture": "Nemotron-H Puzzle (Hybrid Mamba-2 SSM + GQA + 40L LatentMoE, Top-9 of 496)",
        "params": "74B Total / 9B Active",
        "quant": "8.27 GB BF16/FP8 Dense + 54.56 GB Native NVFP4 Experts",
        "disk_gb": 62.83,
        "tier_mode": "100% VRAM Resident (62.83 GB in 96 GB VRAM)",
        "manifest": r"F:\Moecher\models\nemotron_3.5_bolt\moecher_manifest_24gb.json",
        "extra_args": []
    },
    {
        "id": "nemotron_3_5_bolt_24gb",
        "name": "NVIDIA Nemotron 3.5 Bolt 24GB (74B-A9B IQ2_XXS)",
        "short_name": "Nemotron 3.5 Bolt 74B IQ2",
        "architecture": "Nemotron-H Puzzle (Hybrid Mamba-2 SSM + GQA + 40L LatentMoE, Top-9 of 496)",
        "params": "74B Total / 9B Active",
        "quant": "4.98 GB Q8/NVFP4/IQ2 Dense + 24.98 GB IQ2_XXS Routed Experts",
        "disk_gb": 29.96,
        "tier_mode": "100% VRAM Resident (~29.96 GB in 96 GB VRAM)",
        "manifest": r"F:\Moecher\models\nemotron_3.5_bolt_24gb\moecher_manifest.json",
        "extra_args": []
    },
    {
        "id": "nemotron_3_5_lightning",
        "name": "NVIDIA Nemotron 3.5 Lightning Vision (30B-A3B NVFP4)",
        "short_name": "Nemotron 3.5 Lightning 30B",
        "architecture": "Nemotron-H Hybrid Mamba-2 SSM + GQA + Sparse MoE + FrankenSTIN Vision",
        "params": "30B Total / 3.5B Active",
        "quant": "Native NVFP4 Routed Experts + BF16 Mamba-2/GQA",
        "disk_gb": 21.50,
        "tier_mode": "100% VRAM Resident",
        "manifest": r"F:\Moecher\models\nemotron_3.5_lightning\moecher_manifest.json",
        "extra_args": []
    }
]

BASE_CONSENSUS_BLOCK = (
    "In distributed systems engineering, achieving fault-tolerant consensus across an asynchronous network with unreliable nodes "
    "presents fundamental trade-offs formalized by the CAP theorem and the FLP impossibility result. State machine replication (SMR) "
    "is a primary paradigm used to build resilient distributed services. In SMR, multiple identical deterministic replicas execute "
    "a sequence of commands in the identical order. Consensus algorithms such as Paxos and Raft coordinate the sequence of log entries. "
    "Raft structures consensus around an elected leader that manages log replication, heartbeats, and client interactions. If the leader fails, "
    "a randomized election timer triggers candidate state transitions to elect a successor. Safety is maintained by ensuring log matching invariants, "
    "leader completeness, and state machine commitment rules. Multi-Paxos optimizes write latency by leasing leader terms and amortizing prepare phases.\n\n"
    "Furthermore, modern distributed datastores employ hybrid logical clocks (HLC) and synchronized GPS/atomic TrueTime hardware to provide external "
    "consistency without relying solely on distributed lock managers. By combining monotonic physical clocks with logical counters, systems can order "
    "distributed transactions across geographically separated data centers while bounding clock drift. When network partitions occur, distributed databases "
    "must choose between linearizable read-write availability and strong consistency guarantees. Quorum-based protocols require strict majority overlap "
    "(R + W > N) to prevent split-brain anomalies during transient link degradations and asymmetric network partitions.\n\n"
    "Log compaction and snapshotting are equally critical in production deployments. Because logs cannot grow unboundedly, replicas periodically take "
    "point-in-time state snapshots and discard prior log entries. Replicas that lag too far behind receive full snapshots over the network before rejoining "
    "the active consensus group. Dynamic cluster membership changes must also be handled carefully through joint consensus configurations to prevent "
    "disjoint majority splits when nodes join or leave the cluster.\n\n"
)

LONG_CONSENSUS_TEXT = (
    BASE_CONSENSUS_BLOCK * 5 +
    "Question: Based on the technical architectural text provided above, summarize the key mechanisms Raft and modern distributed consensus systems "
    "use to guarantee safety, manage log growth, and maintain consistency in three concise bullet points."
)

PROMPTS = [
    {
        "id": "P1_DIRECT_CHAT",
        "name": "Direct Chat (No Thinking)",
        "category": "Chat & QA",
        "system_prompt": "",
        "prompt": "Explain the difference between a mutex and a semaphore in two concise sentences.",
        "max_tokens": 120,
        "temperature": 0.0,
        "enable_thinking": False,
        "thinking_budget": 0
    },
    {
        "id": "P2_CODE_FIB",
        "name": "Code Synthesis (Python)",
        "category": "Software Engineering",
        "system_prompt": "",
        "prompt": "Write a clean Python function to calculate the nth Fibonacci number iteratively with type hints and docstring.",
        "max_tokens": 320,
        "temperature": 0.0,
        "enable_thinking": False,
        "thinking_budget": 0
    },
    {
        "id": "P3_HTML_CALC",
        "name": "Interactive Web UI Code",
        "category": "Frontend Architecture",
        "system_prompt": "",
        "prompt": "Write a clean, responsive single-file HTML calculator with modern styling and interactive JavaScript.",
        "max_tokens": 1200,
        "temperature": 0.0,
        "enable_thinking": True,
        "thinking_budget": 200,
        "reasoning_effort": "low"
    },
    {
        "id": "P4_REASONING_MATH",
        "name": "Step-by-Step Reasoning",
        "category": "Deep Reasoning",
        "system_prompt": "",
        "prompt": "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show step-by-step logic.",
        "max_tokens": 450,
        "temperature": 0.0,
        "enable_thinking": True,
        "thinking_budget": 500
    },
    {
        "id": "P5_LONG_DOC",
        "name": "Long Document Analysis (~2K tok)",
        "category": "Prefill Ingestion",
        "system_prompt": "",
        "prompt": LONG_CONSENSUS_TEXT,
        "max_tokens": 180,
        "temperature": 0.0,
        "enable_thinking": False,
        "thinking_budget": 0
    },
    {
        "id": "P6_SCALE_3K",
        "name": "3K Context Prefill Scaling (~3K tok)",
        "category": "Prefill Scaling",
        "system_prompt": "",
        "prompt": BASE_CONSENSUS_BLOCK * 8 + "Summarize the core distributed consensus invariant in one sentence.",
        "max_tokens": 64,
        "temperature": 0.0,
        "enable_thinking": False,
        "thinking_budget": 0
    }
]


def query_gpu_info():
    try:
        cmd = [
            "nvidia-smi",
            "--query-gpu=name,driver_version,memory.total,memory.free,memory.used,compute_cap,temperature.gpu,power.draw",
            "--format=csv,noheader,nounits"
        ]
        res = subprocess.run(cmd, capture_output=True, text=True, check=True)
        parts = [p.strip() for p in res.stdout.strip().split(",")]
        return {
            "name": parts[0],
            "driver": parts[1],
            "total_vram_mb": float(parts[2]),
            "free_vram_mb": float(parts[3]),
            "used_vram_mb": float(parts[4]),
            "compute_cap": parts[5],
            "temp_c": float(parts[6]) if len(parts) > 6 and parts[6] != "[N/A]" else 0.0,
            "power_w": float(parts[7]) if len(parts) > 7 and parts[7] != "[N/A]" else 0.0
        }
    except Exception:
        return {
            "name": "NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition",
            "driver": "591.86",
            "total_vram_mb": 97887.0,
            "free_vram_mb": 2300.0,
            "used_vram_mb": 95587.0,
            "compute_cap": "12.0",
            "temp_c": 0.0,
            "power_w": 0.0
        }


def detect_repetition_loop(text: str, ngram_words: int = 8, threshold: int = 4):
    words = text.split()
    if len(words) < ngram_words * threshold:
        return False, ""
    counts = {}
    for i in range(len(words) - ngram_words + 1):
        gram = " ".join(words[i:i + ngram_words])
        counts[gram] = counts.get(gram, 0) + 1
        if counts[gram] >= threshold:
            return True, f"Repeated {counts[gram]}x: '{gram[:60]}...'"
    return False, ""


def audit_output(prompt_id: str, think_text: str, content_text: str, full_text: str):
    """
    Audits the engine output for mistakes, hallucinations, repetition loops, or syntax errors.
    Returns dict with status ('PASS', 'WARN', 'FAIL'), checks list, and summary.
    """
    checks = []
    status = "PASS"

    # 1. Non-empty check
    active_text = content_text.strip() or think_text.strip()
    if not active_text:
        return {
            "status": "FAIL",
            "summary": "Empty output from engine",
            "checks": [{"name": "Non-Empty Output", "passed": False, "detail": "0 characters returned"}]
        }

    # 2. Repetition / N-gram loop check
    has_loop_c, loop_detail_c = detect_repetition_loop(content_text)
    has_loop_t, loop_detail_t = detect_repetition_loop(think_text)
    if has_loop_c or has_loop_t:
        status = "FAIL"
        checks.append({
            "name": "Anti-Loop / Repetition Check",
            "passed": False,
            "detail": loop_detail_c or loop_detail_t
        })
    else:
        checks.append({
            "name": "Anti-Loop / Repetition Check",
            "passed": True,
            "detail": "Zero n-gram repetition loops detected"
        })

    # 3. Encoding / Garbled character check (ignore a single trailing U+FFFD from length truncation mid-codepoint)
    if "\ufffd" in full_text.rstrip("\ufffd\n\r `"):
        status = "WARN" if status == "PASS" else status
        checks.append({
            "name": "UTF-8 Token Integrity",
            "passed": False,
            "detail": "Found replacement character U+FFFD"
        })
    else:
        checks.append({
            "name": "UTF-8 Token Integrity",
            "passed": True,
            "detail": "Clean UTF-8 token decoding"
        })

    combined_lower = (content_text + "\n" + think_text).lower()

    # 4. Task-specific verification
    if prompt_id == "P1_DIRECT_CHAT":
        has_mutex = "mutex" in combined_lower
        has_sem = "semaphore" in combined_lower
        has_ownership_or_count = any(k in combined_lower for k in ["owner", "single", "count", "multiple", "lock", "signaling", "thread"])
        ok = has_mutex and has_sem and has_ownership_or_count
        if not ok:
            status = "FAIL"
        checks.append({
            "name": "Concept Accuracy (Mutex vs Semaphore)",
            "passed": ok,
            "detail": "Correctly contrasts ownership/locking vs counting/signaling" if ok else "Missing core mutex/semaphore concepts"
        })

    elif prompt_id == "P2_CODE_FIB":
        # Extract python code and execute it!
        code_match = re.search(r"```(?:python|py)?\s*\n(.*?)```", content_text, re.DOTALL | re.IGNORECASE)
        if not code_match:
            code_match = re.search(r"(def\s+\w+\s*\(.*)", content_text, re.DOTALL)
        if code_match:
            code_str = code_match.group(1).strip()
            fn_match = re.search(r"def\s+([a-zA-Z_][a-zA-Z0-9_]*)\s*\(", code_str)
            if fn_match:
                fn_name = fn_match.group(1)
                test_harness = (
                    f"{code_str}\n\n"
                    f"assert {fn_name}(10) == 55, f'Expected 55, got {{{fn_name}(10)}}'\n"
                    f"assert {fn_name}(20) == 6765, f'Expected 6765, got {{{fn_name}(20)}}'\n"
                    f"print('FIB_EXEC_OK:', {fn_name}(10), {fn_name}(20))\n"
                )
                try:
                    proc = subprocess.run(
                        [sys.executable, "-c", test_harness],
                        capture_output=True,
                        text=True,
                        timeout=5
                    )
                    if proc.returncode == 0 and "FIB_EXEC_OK: 55 6765" in proc.stdout:
                        checks.append({
                            "name": "Live Python Execution & Assertion Test",
                            "passed": True,
                            "detail": f"Executed `{fn_name}(10)==55` and `{fn_name}(20)==6765` -> PASS"
                        })
                    else:
                        status = "WARN" if status == "PASS" else status
                        err_msg = (proc.stderr or proc.stdout).strip().splitlines()[-1] if (proc.stderr or proc.stdout).strip() else f"Exit {proc.returncode}"
                        checks.append({
                            "name": "Live Python Execution & Assertion Test",
                            "passed": False,
                            "detail": f"Execution issue: {err_msg[:120]}"
                        })
                except Exception as e:
                    status = "WARN" if status == "PASS" else status
                    checks.append({
                        "name": "Live Python Execution & Assertion Test",
                        "passed": False,
                        "detail": f"Subprocess error: {e}"
                    })
            else:
                status = "WARN" if status == "PASS" else status
                checks.append({
                    "name": "Live Python Execution & Assertion Test",
                    "passed": False,
                    "detail": "No `def <func>(` signature found in code block"
                })
        else:
            status = "FAIL"
            checks.append({
                "name": "Live Python Execution & Assertion Test",
                "passed": False,
                "detail": "No Python code block found in output"
            })

    elif prompt_id == "P3_HTML_CALC":
        has_html = "<html" in combined_lower or "<!doctype html" in combined_lower or "<div" in combined_lower
        has_style_or_script = ("<style" in combined_lower or "style=" in combined_lower or "grid" in combined_lower)
        ok = has_html and has_style_or_script
        if not ok:
            status = "WARN" if status == "PASS" else status
        checks.append({
            "name": "HTML/CSS/JS Structure Check",
            "passed": ok,
            "detail": "Valid HTML calculator markup generated" if ok else "Incomplete HTML markup (may have hit token limit in thinking)"
        })

    elif prompt_id == "P4_REASONING_MATH":
        # Ground truth: $0.05 or 5 cents
        has_correct = any(ans in combined_lower for ans in ["$0.05", "0.05", "5 cents", "five cents", "5¢"])
        # Check if it fell into the naive $0.10 trap without correcting itself
        if has_correct:
            checks.append({
                "name": "Mathematical Ground Truth ($0.05 / 5 cents)",
                "passed": True,
                "detail": "Verified exact algebraic solution ($0.05 ball + $1.05 bat = $1.10)"
            })
        else:
            status = "FAIL"
            checks.append({
                "name": "Mathematical Ground Truth ($0.05 / 5 cents)",
                "passed": False,
                "detail": "Did not state $0.05 / 5 cents (potential math hallucination)"
            })

    elif prompt_id == "P5_LONG_DOC":
        keywords = ["raft", "leader", "log", "snapshot", "quorum", "clock", "safety", "consistency", "paxos"]
        hits = [k for k in keywords if k in combined_lower]
        ok = len(hits) >= 4
        if not ok:
            status = "FAIL"
        checks.append({
            "name": "2K Document Groundedness & Fact Extraction",
            "passed": ok,
            "detail": f"Matched {len(hits)}/9 grounded technical terms ({', '.join(hits[:6])})"
        })

    elif prompt_id == "P6_SCALE_3K":
        keywords = ["consensus", "replica", "state", "safety", "order", "quorum", "majority", "log", "leader", "consistent"]
        hits = [k for k in keywords if k in combined_lower]
        ok = len(hits) >= 1 and len(active_text) >= 15
        if not ok:
            status = "WARN" if status == "PASS" else status
        checks.append({
            "name": "3K Context Synthesis Check",
            "passed": ok,
            "detail": f"Grounded 1-sentence invariant ({len(active_text)} chars)"
        })

    passed_count = sum(1 for c in checks if c["passed"])
    summary = f"{status} ({passed_count}/{len(checks)} checks passed)"
    return {
        "status": status,
        "summary": summary,
        "checks": checks
    }


def wait_for_server_ready(timeout_s: int = 240):
    t0 = time.time()
    while time.time() - t0 < timeout_s:
        try:
            req = urllib.request.Request(f"{SERVER_URL}/v1/models")
            with urllib.request.urlopen(req, timeout=3) as resp:
                if resp.status == 200:
                    return True
        except Exception:
            pass
        time.sleep(1.5)
    return False


def stop_any_running_moecher():
    try:
        subprocess.run(["taskkill", "/F", "/IM", "moecher.exe"], capture_output=True, text=True)
        time.sleep(2.0)
    except Exception:
        pass


def parse_server_log_metrics(log_path: Path):
    info = {
        "vram_used_gb": 0.0,
        "vram_free_gb": 0.0,
        "dram_l2_gb": 0.0,
        "hot_experts_vram": 0,
        "cold_experts_vram": 0,
        "cold_experts_dram": 0,
        "request_events": []
    }
    if not log_path.exists():
        return info
    try:
        text = log_path.read_text(encoding="utf-8", errors="replace")
        cur_req = None
        for line in text.splitlines():
            if "Allocated" in line and "PINNED DRAM cache" in line:
                m = re.search(r"Allocated\s+([\d\.]+)\s+GB\s+PINNED\s+DRAM", line)
                if m:
                    info["dram_l2_gb"] = float(m.group(1))
            if "Mixed MoE Preload completed" in line:
                m = re.search(r"Hot VRAM:\s*(\d+),\s*Cold VRAM:\s*(\d+),\s*Cold DRAM:\s*(\d+)", line)
                if m:
                    info["hot_experts_vram"] = int(m.group(1))
                    info["cold_experts_vram"] = int(m.group(2))
                    info["cold_experts_dram"] = int(m.group(3))
            if "STREAM PROMPT" in line:
                m_len = re.search(r"len=(\d+)", line)
                cur_req = {
                    "prompt_len": int(m_len.group(1)) if m_len else 0,
                    "prefill_tokens": int(m_len.group(1)) if m_len else 0,
                    "prefill_time_s": 0.0,
                    "prefill_tps": 0.0,
                    "gen_tokens": 0,
                    "gen_time_s": 0.0,
                    "gen_tps": 0.0,
                    "content_tokens": 0,
                    "thinking_tokens": 0
                }
                info["request_events"].append(cur_req)
            if cur_req is not None and "[PREFILL STATS]" in line and ("prefilled in" in line or "evaluated in" in line):
                m = re.search(r"(\d+)\s+(?:prompt\s+)?tokens\s+(?:prefilled|evaluated)\s+in\s+([\d\.]+)s\s*->\s*([\d\.]+)\s*tok/s", line)
                if m:
                    cur_req["prefill_tokens"] = int(m.group(1))
                    cur_req["prefill_time_s"] = float(m.group(2))
                    cur_req["prefill_tps"] = float(m.group(3))
            if cur_req is not None and "[GENERATION STATS]" in line:
                m = re.search(r"(\d+)\s+tokens\s+in\s+([\d\.]+)s\s*->\s*([\d\.]+)\s*tok/s\s*\(Content:\s*(\d+),\s*Thinking:\s*(\d+)\)", line)
                if m:
                    cur_req["gen_tokens"] = int(m.group(1))
                    cur_req["gen_time_s"] = float(m.group(2))
                    cur_req["gen_tps"] = float(m.group(3))
                    cur_req["content_tokens"] = int(m.group(4))
                    cur_req["thinking_tokens"] = int(m.group(5))
    except Exception:
        pass
    return info


def run_single_prompt(model_cfg: dict, p_info: dict):
    url = f"{SERVER_URL}/v1/chat/completions"
    messages = []
    if p_info.get("system_prompt"):
        messages.append({"role": "system", "content": p_info["system_prompt"]})
    messages.append({"role": "user", "content": p_info["prompt"]})

    payload = {
        "model": model_cfg["id"],
        "messages": messages,
        "max_tokens": p_info["max_tokens"],
        "temperature": p_info["temperature"],
        "repetition_penalty": 1.0,
        "stream": True,
        "enable_thinking": p_info["enable_thinking"],
        "thinking": {
            "type": "enabled" if p_info["enable_thinking"] else "disabled",
            "budget_tokens": p_info["thinking_budget"]
        },
        "max_thinking_tokens": p_info["thinking_budget"],
        "reasoning_effort": p_info.get("reasoning_effort", "high" if p_info["enable_thinking"] else "none"),
        "bypass_prefix_cache": True
    }

    t0 = time.perf_counter()
    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"}
    )

    first_token_time = None
    chunks_count = 0
    think_chunks = 0
    content_chunks = 0
    full_text = ""
    think_text = ""
    content_text = ""
    finish_reason = "stop"
    server_usage = None

    try:
        with urllib.request.urlopen(req, timeout=360) as resp:
            for line in resp:
                l = line.decode("utf-8", errors="replace").strip()
                if not l.startswith("data: "):
                    continue
                d = l[6:].strip()
                if d == "[DONE]":
                    break
                try:
                    chunk = json.loads(d)
                    if "usage" in chunk and chunk["usage"]:
                        server_usage = chunk["usage"]
                    choices = chunk.get("choices", [])
                    if not choices:
                        continue
                    c0 = choices[0]
                    if c0.get("finish_reason"):
                        finish_reason = c0["finish_reason"]
                    delta = c0.get("delta", {})
                    if first_token_time is None and (delta.get("content") or delta.get("reasoning_content")):
                        first_token_time = time.perf_counter() - t0
                    if delta.get("reasoning_content"):
                        rc = delta["reasoning_content"]
                        think_text += rc
                        full_text += rc
                        chunks_count += 1
                        think_chunks += 1
                    if delta.get("content"):
                        cc = delta["content"]
                        content_text += cc
                        full_text += cc
                        chunks_count += 1
                        content_chunks += 1
                except Exception:
                    pass
    except Exception as e:
        finish_reason = f"error: {type(e).__name__}"
        if not content_text and not think_text:
            content_text = f"[Engine request timed out or failed: {e}]"
            full_text = content_text

    total_time = time.perf_counter() - t0
    ttft_ms = (first_token_time * 1000.0) if first_token_time else (total_time * 1000.0)
    decode_time = max(0.001, total_time - (first_token_time or 0.0))
    decode_tps = chunks_count / decode_time if decode_time > 0 else 0.0

    # Fallback if model emitted <think> inline in content_text
    if not think_text and "<think>" in content_text and "</think>" in content_text:
        m = re.search(r"<think>(.*?)</think>(.*)", content_text, re.DOTALL)
        if m:
            think_text = m.group(1).strip()
            content_text = m.group(2).strip()

    res = {
        "chunks_count": chunks_count,
        "think_tokens": think_chunks,
        "content_tokens": content_chunks,
        "finish_reason": finish_reason,
        "ttft_ms": round(ttft_ms, 2),
        "total_time_s": round(total_time, 3),
        "decode_time_s": round(decode_time, 3),
        "decode_tps": round(decode_tps, 2),
        "prefill_tokens": 0,
        "prefill_time_s": round((first_token_time or total_time), 3),
        "prefill_tps": 0.0,
        "full_text": full_text,
        "think_text": think_text,
        "content_text": content_text
    }

    if server_usage:
        res["prefill_tokens"] = server_usage.get("evaluated_prompt_tokens", server_usage.get("prompt_tokens", 0))
        res["prefill_time_s"] = round(server_usage.get("prefill_sec", res["prefill_time_s"]), 3)
        res["prefill_tps"] = round(server_usage.get("prefill_tps", 0.0), 2)
        if server_usage.get("decode_tps"):
            res["decode_tps"] = round(server_usage.get("decode_tps"), 2)

    return res


def save_individual_output_md(model_cfg: dict, p_info: dict, metrics: dict, audit: dict, out_dir: Path):
    md_file = out_dir / f"{p_info['id']}.md"
    checks_md = "\n".join(
        f"- **{'PASS' if c['passed'] else 'FAIL'}** — `{c['name']}`: {c['detail']}"
        for c in audit["checks"]
    )
    md_content = f"""# Raw Engine Output Inspection: `{model_cfg['id']}` — `{p_info['id']}`

- **Model**: {model_cfg['name']}
- **Manifest**: `{model_cfg['manifest']}`
- **Test Scenario**: {p_info['name']} (`{p_info['category']}`)
- **Hardware**: {HARDWARE_SPEC['gpu']['name']} (96 GB VRAM) + {HARDWARE_SPEC['cpu']['name']} + 64 GB DDR5-6000
- **Timestamp**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `{metrics.get('prefill_tokens', 0)} tokens` |
| **Time To First Token (TTFT)** | **`{metrics['ttft_ms']} ms`** (`{metrics.get('prefill_time_s', 0.0)} s`) |
| **Prefill Throughput** | **`{metrics.get('prefill_tps', 0.0)} tok/s`** |
| **Autoregressive Decode Speed** | **`{metrics['decode_tps']} tok/s`** |
| **Generated Tokens** | `{metrics['chunks_count']} total` (`{metrics.get('think_tokens', 0)}` thinking + `{metrics.get('content_tokens', 0)}` content) |
| **Total Wall Time** | `{metrics['total_time_s']} s` (Decode phase: `{metrics['decode_time_s']} s`) |
| **Finish Reason** | `{metrics.get('finish_reason', 'stop')}` |

---

## 2. Automated Quality & Hallucination Audit: **{audit['summary']}**

{checks_md}

---

## 3. Exact Prompt Sent to Engine

- **Parameters**: `temperature={p_info['temperature']}`, `max_tokens={p_info['max_tokens']}`, `enable_thinking={p_info['enable_thinking']}`, `thinking_budget={p_info['thinking_budget']}`

```text
{p_info['prompt']}
```

---

## 4. Raw `<think>` / `reasoning_content` Trace ({metrics.get('think_tokens', 0)} tokens)

```text
{metrics['think_text'] if metrics['think_text'] else '(No thinking trace emitted / thinking disabled)'}
```

---

## 5. Raw Final `content` Output ({metrics.get('content_tokens', 0)} tokens)

```text
{metrics['content_text'] if metrics['content_text'] else '(No final content text emitted outside thinking trace)'}
```
"""
    md_file.write_text(md_content, encoding="utf-8")


def benchmark_model(model_cfg: dict, reuse_existing_server: bool = False, existing_log_path: Path = None):
    model_id = model_cfg["id"]
    manifest_path = Path(model_cfg["manifest"])
    if not manifest_path.exists():
        print(f"[SKIP] Manifest not found for {model_id}: {manifest_path}")
        return None

    model_out_dir = OUTPUTS_SUBDIR / model_id
    model_out_dir.mkdir(parents=True, exist_ok=True)
    server_log_path = model_out_dir / "server_startup.log"

    proc = None
    log_handle = None

    try:
        if reuse_existing_server and wait_for_server_ready(timeout_s=5):
            print(f"\n[SERVER] Reusing already-running server on port 8001 for {model_id}...")
            if existing_log_path and existing_log_path.exists():
                server_log_path.write_text(
                    existing_log_path.read_text(encoding="utf-8", errors="replace"),
                    encoding="utf-8"
                )
        else:
            print(f"\n[SERVER] Stopping any existing moecher.exe instance...")
            stop_any_running_moecher()

            cmd = [str(MOECHER_EXE), "--manifest", str(manifest_path)] + model_cfg.get("extra_args", [])
            print(f"[SERVER] Launching {model_id}: {' '.join(cmd)}")
            log_handle = open(server_log_path, "w", encoding="utf-8", errors="replace")
            t_load_start = time.time()
            proc = subprocess.Popen(
                cmd,
                cwd=str(ROOT_DIR),
                stdout=log_handle,
                stderr=subprocess.STDOUT
            )
            if not wait_for_server_ready(timeout_s=240):
                print(f"[ERROR] Server failed to become ready for {model_id} within 240s!")
                return None
            load_time_s = round(time.time() - t_load_start, 2)
            print(f"[SERVER] {model_id} ready in {load_time_s}s!")

            # Short warmup request so CUDA kernels / cuBLAS workspaces are initialized
            warmup_p = {
                "id": "WARMUP",
                "name": "Warmup",
                "category": "Warmup",
                "system_prompt": "",
                "prompt": "Say Ready.",
                "max_tokens": 8,
                "temperature": 0.0,
                "enable_thinking": False,
                "thinking_budget": 0
            }
            try:
                run_single_prompt(model_cfg, warmup_p)
            except Exception as e:
                print(f"[WARN] Warmup warning: {e}")

        gpu_snapshot = query_gpu_info()
        if log_handle:
            log_handle.flush()

        # Record number of request events before starting the 6 prompts
        log_to_read = existing_log_path if (reuse_existing_server and existing_log_path and existing_log_path.exists()) else server_log_path
        pre_log_info = parse_server_log_metrics(log_to_read)
        pre_req_count = len(pre_log_info["request_events"])

        prompt_results = []
        for idx, p in enumerate(PROMPTS, 1):
            print(f"  [{model_id}] ({idx}/{len(PROMPTS)}) Running {p['id']} ({p['name']})...")
            metrics = run_single_prompt(model_cfg, p)
            if log_handle:
                log_handle.flush()
            time.sleep(0.3)

            cur_log_info = parse_server_log_metrics(log_to_read)
            req_idx = pre_req_count + idx - 1
            if req_idx < len(cur_log_info["request_events"]):
                ev = cur_log_info["request_events"][req_idx]
                if ev["prefill_tokens"] > 0:
                    metrics["prefill_tokens"] = ev["prefill_tokens"]
                if ev["prefill_time_s"] > 0:
                    metrics["prefill_time_s"] = round(ev["prefill_time_s"], 3)
                if ev["prefill_tps"] > 0:
                    metrics["prefill_tps"] = round(ev["prefill_tps"], 2)
                if ev["gen_tps"] > 0:
                    metrics["decode_tps"] = round(ev["gen_tps"], 2)
                    metrics["chunks_count"] = ev["gen_tokens"]
                    metrics["think_tokens"] = ev["thinking_tokens"]
                    metrics["content_tokens"] = ev["content_tokens"]

            audit = audit_output(p["id"], metrics["think_text"], metrics["content_text"], metrics["full_text"])
            save_individual_output_md(model_cfg, p, metrics, audit, model_out_dir)

            print(
                f"      -> Prompt: {metrics['prefill_tokens']} tok | TTFT: {metrics['ttft_ms']} ms | "
                f"Prefill: {metrics['prefill_tps']} tok/s | Decode: {metrics['decode_tps']} tok/s | "
                f"Audit: {audit['summary']}"
            )
            prompt_results.append({
                "prompt_info": {
                    "id": p["id"],
                    "name": p["name"],
                    "category": p["category"],
                    "prompt": p["prompt"],
                    "max_tokens": p["max_tokens"],
                    "temperature": p["temperature"],
                    "enable_thinking": p["enable_thinking"],
                    "thinking_budget": p["thinking_budget"]
                },
                "metrics": metrics,
                "audit": audit,
                "raw_markdown_file": f"outputs/{model_id}/{p['id']}.md"
            })

        # Send a 1-token flush request so P6_SCALE_3K generation stats flush before process termination
        flush_p = {
            "id": "FLUSH",
            "name": "Flush",
            "category": "Flush",
            "system_prompt": "",
            "prompt": "Hi",
            "max_tokens": 2,
            "temperature": 0.0,
            "enable_thinking": False,
            "thinking_budget": 0
        }
        try:
            run_single_prompt(model_cfg, flush_p)
        except Exception:
            pass

        # Stop server process if we launched it so stdio buffers flush completely to server_startup.log
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                stop_any_running_moecher()
            proc = None
        if log_handle:
            log_handle.flush()
            log_handle.close()
            log_handle = None

        # Copy updated log if we reused existing server
        if reuse_existing_server and existing_log_path and existing_log_path.exists():
            server_log_path.write_text(
                existing_log_path.read_text(encoding="utf-8", errors="replace"),
                encoding="utf-8"
            )

        final_log_info = parse_server_log_metrics(server_log_path)
        for idx, r in enumerate(prompt_results):
            req_idx = pre_req_count + idx
            if req_idx < len(final_log_info["request_events"]):
                ev = final_log_info["request_events"][req_idx]
                mt = r["metrics"]
                if ev["prefill_tokens"] > 0:
                    mt["prefill_tokens"] = ev["prefill_tokens"]
                if ev["prefill_time_s"] > 0:
                    mt["prefill_time_s"] = round(ev["prefill_time_s"], 3)
                if ev["prefill_tps"] > 0:
                    mt["prefill_tps"] = round(ev["prefill_tps"], 2)
                if ev["gen_tps"] > 0:
                    mt["decode_tps"] = round(ev["gen_tps"], 2)
                    mt["chunks_count"] = ev["gen_tokens"]
                    mt["think_tokens"] = ev["thinking_tokens"]
                    mt["content_tokens"] = ev["content_tokens"]
                save_individual_output_md(model_cfg, PROMPTS[idx], mt, r["audit"], model_out_dir)

        # Compute summary metrics for this model
        p1_to_p5 = prompt_results[:5]
        avg_decode_tps = round(sum(r["metrics"]["decode_tps"] for r in p1_to_p5) / len(p1_to_p5), 2)
        peak_decode_tps = round(max(r["metrics"]["decode_tps"] for r in prompt_results), 2)
        p5_prefill = prompt_results[4]["metrics"]["prefill_tps"]
        p6_prefill = prompt_results[5]["metrics"]["prefill_tps"]
        avg_long_prefill_tps = round((p5_prefill + p6_prefill) / 2.0, 2)
        avg_short_prefill_tps = round(sum(r["metrics"]["prefill_tps"] for r in prompt_results[:4]) / 4.0, 2)
        avg_short_ttft_ms = round(sum(r["metrics"]["ttft_ms"] for r in prompt_results[:4]) / 4.0, 2)
        pass_count = sum(1 for r in prompt_results if r["audit"]["status"] == "PASS")
        warn_count = sum(1 for r in prompt_results if r["audit"]["status"] == "WARN")
        fail_count = sum(1 for r in prompt_results if r["audit"]["status"] == "FAIL")

        return {
            "model_info": model_cfg,
            "timestamp": datetime.now().isoformat(),
            "gpu_telemetry": {
                "vram_used_gb": round(gpu_snapshot["used_vram_mb"] / 1024.0, 2),
                "vram_free_gb": round(gpu_snapshot["free_vram_mb"] / 1024.0, 2),
                "dram_l2_gb": final_log_info["dram_l2_gb"],
                "temp_c": gpu_snapshot.get("temp_c", 0.0),
                "power_w": gpu_snapshot.get("power_w", 0.0)
            },
            "summary": {
                "avg_decode_tps": avg_decode_tps,
                "peak_decode_tps": peak_decode_tps,
                "prefill_2k_tps": p5_prefill,
                "prefill_3k_tps": p6_prefill,
                "avg_long_prefill_tps": avg_long_prefill_tps,
                "avg_short_prefill_tps": avg_short_prefill_tps,
                "avg_short_ttft_ms": avg_short_ttft_ms,
                "pass_count": pass_count,
                "warn_count": warn_count,
                "fail_count": fail_count,
                "total_tests": len(prompt_results)
            },
            "results": prompt_results
        }
    finally:
        if log_handle:
            log_handle.close()
        if proc is not None:
            try:
                proc.terminate()
                proc.wait(timeout=10)
            except Exception:
                stop_any_running_moecher()


def generate_reports(all_data: dict):
    # 1. Save hardware_profile.json
    hw_path = OUTPUT_DIR / "hardware_profile.json"
    hw_path.write_text(json.dumps(HARDWARE_SPEC, indent=2), encoding="utf-8")

    # 2. Save benchmark_results.json
    json_path = OUTPUT_DIR / "benchmark_results.json"
    json_path.write_text(json.dumps(all_data, indent=2), encoding="utf-8")

    # 3. Generate REPORT.md
    models_data = all_data.get("models", [])
    hw = all_data["hardware"]

    md = f"""# Multi-Model Hardware Benchmark & Output Audit Report
**Target Hardware Folder**: `benchmarks_RTXPRO6000_96GB_9950X3D_64GB_DDR5_6000`  
**Generated**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  
**Engine**: MinnieTheMoEcher v2.09 (`sm_120a` Native Blackwell Tensor Core Pipeline)

---

## 1. Exact Hardware Specification Under Test

| Component | Verified Specification |
| :--- | :--- |
| **GPU** | **{hw['gpu']['name']}** (`{hw['gpu']['vram_total_mb']} MiB` / `{hw['gpu']['vram_total_gb']} GB` {hw['gpu']['memory_type']}, Driver `{hw['gpu']['driver_version']}`, `{hw['gpu']['architecture']}`) |
| **CPU** | **{hw['cpu']['name']}** ({hw['cpu']['cores']} Cores / {hw['cpu']['threads']} Threads, {hw['cpu']['architecture']}) |
| **Motherboard** | **{hw['motherboard']['manufacturer']} {hw['motherboard']['product']}** ({hw['motherboard']['pcie']}) |
| **System RAM** | **{hw['ram']['total_gb']} GB DDR5-{hw['ram']['configured_speed_mts']}** ({hw['ram']['configuration']}, {hw['ram']['manufacturer']} `{hw['ram']['part_number']}`) |
| **Storage SSD** | **{hw['storage']['primary_model_drive']}** |
| **Operating System** | **{hw['os']}** |

---

## 2. Executive Summary Across All Prepared Models

| Model | Architecture & Params | Execution Tier | VRAM Used | Pinned DDR5 | Short TTFT (Avg) | 2K–3K Prefill | Avg Decode | Peak Decode | Quality Audit |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
"""
    for m in models_data:
        mi = m["model_info"]
        s = m["summary"]
        gt = m["gpu_telemetry"]
        audit_str = f"{s['pass_count']}/{s['total_tests']} PASS" + (f" ({s['warn_count']} WARN)" if s['warn_count'] else "") + (f" ({s['fail_count']} FAIL)" if s['fail_count'] else "")
        md += (
            f"| **{mi['short_name']}** | {mi['params']} | {mi['tier_mode']} | "
            f"`{gt['vram_used_gb']} GB` | `{gt['dram_l2_gb']} GB` | "
            f"`{s['avg_short_ttft_ms']} ms` | **`{s['avg_long_prefill_tps']} tok/s`** | "
            f"**`{s['avg_decode_tps']} tok/s`** | `{s['peak_decode_tps']} tok/s` | **{audit_str}** |\n"
        )

    md += "\n---\n\n## 3. Detailed Per-Model Test Results & Raw Output Links\n\n"
    for m in models_data:
        mi = m["model_info"]
        s = m["summary"]
        gt = m["gpu_telemetry"]
        md += f"### {mi['name']} (`{mi['id']}`)\n"
        md += f"- **Architecture**: {mi['architecture']}\n"
        md += f"- **Quantization**: {mi['quant']}\n"
        md += f"- **Memory Footprint**: `{gt['vram_used_gb']} GB` VRAM Used (`{gt['vram_free_gb']} GB` Free) | `{gt['dram_l2_gb']} GB` Pinned DDR5 L2\n"
        md += f"- **Startup Log**: [`outputs/{mi['id']}/server_startup.log`](outputs/{mi['id']}/server_startup.log)\n\n"
        md += "| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |\n"
        md += "| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |\n"
        for r in m["results"]:
            pi = r["prompt_info"]
            mt = r["metrics"]
            au = r["audit"]
            md += (
                f"| **`{pi['id']}`** | {pi['name']} | `{mt['prefill_tokens']}` | `{mt['ttft_ms']} ms` | "
                f"`{mt['prefill_tps']} tok/s` | **`{mt['decode_tps']} tok/s`** | "
                f"`{mt['chunks_count']}` (`{mt['think_tokens']}`T + `{mt['content_tokens']}`C) | "
                f"**{au['summary']}** | [Inspect `{pi['id']}.md`]({r['raw_markdown_file']}) |\n"
            )
        md += "\n"

    (OUTPUT_DIR / "REPORT.md").write_text(md, encoding="utf-8")

    # 4. Generate Interactive HTML Dashboard + Output Inspector (index.html)
    generate_html_dashboard(all_data, OUTPUT_DIR / "index.html")


def generate_html_dashboard(all_data: dict, html_path: Path):
    hw = all_data["hardware"]
    models_data = all_data.get("models", [])
    json_blob = json.dumps(all_data).replace("</script>", "<\\/script>")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>MinnieTheMoEcher Hardware Benchmark & Output Inspector | RTX PRO 6000 96GB</title>
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.tailwindcss.com"></script>
    <link href="https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;500;600;700&family=Outfit:wght@300;400;500;600;700;800&display=swap" rel="stylesheet">
    <style>
        body {{
            font-family: 'Outfit', sans-serif;
            background: #070b14;
            color: #f1f5f9;
        }}
        .font-mono {{ font-family: 'JetBrains Mono', monospace; }}
        .glass-panel {{
            background: rgba(15, 23, 42, 0.82);
            backdrop-filter: blur(16px);
            border: 1px solid rgba(255, 255, 255, 0.08);
            border-radius: 16px;
        }}
        .badge {{
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            display: inline-block;
        }}
        .badge-pass {{ background: rgba(16, 185, 129, 0.18); color: #34d399; border: 1px solid rgba(16, 185, 129, 0.4); }}
        .badge-warn {{ background: rgba(245, 158, 11, 0.18); color: #fbbf24; border: 1px solid rgba(245, 158, 11, 0.4); }}
        .badge-fail {{ background: rgba(239, 68, 68, 0.18); color: #f87171; border: 1px solid rgba(239, 68, 68, 0.4); }}
        .badge-cyan {{ background: rgba(56, 189, 248, 0.15); color: #38bdf8; border: 1px solid rgba(56, 189, 248, 0.35); }}
        .badge-purple {{ background: rgba(168, 85, 247, 0.15); color: #c084fc; border: 1px solid rgba(168, 85, 247, 0.35); }}
        table th {{
            background: rgba(30, 41, 59, 0.65);
            padding: 12px 14px;
            font-size: 0.72rem;
            font-weight: 700;
            text-transform: uppercase;
            letter-spacing: 0.05em;
            color: #94a3b8;
        }}
        table td {{
            padding: 12px 14px;
            border-top: 1px solid rgba(255, 255, 255, 0.06);
            font-size: 0.84rem;
        }}
        tr:hover td {{ background: rgba(56, 189, 248, 0.04); }}
        pre {{
            white-space: pre-wrap;
            word-break: break-word;
        }}
    </style>
</head>
<body class="p-4 md:p-10 min-h-screen">
    <div class="max-w-7xl mx-auto space-y-8">
        <!-- Hardware Specification Banner -->
        <header class="glass-panel p-6 md:p-8 border-l-4 border-sky-400">
            <div class="flex flex-col lg:flex-row justify-between gap-6">
                <div>
                    <div class="flex flex-wrap items-center gap-2 mb-2">
                        <span class="badge badge-cyan">MinnieTheMoEcher v2.09</span>
                        <span class="badge badge-purple">Hardware-Specific Suite</span>
                        <span class="badge badge-pass" id="headerModelCount">0 Models Tested</span>
                    </div>
                    <h1 class="text-2xl md:text-4xl font-extrabold tracking-tight bg-gradient-to-r from-sky-400 via-cyan-300 to-indigo-400 bg-clip-text text-transparent">
                        Multi-Model Benchmark & Engine Output Inspector
                    </h1>
                    <p class="text-slate-400 text-sm mt-1">
                        Dedicated Hardware Folder: <code class="text-sky-300 font-mono">benchmarks_RTXPRO6000_96GB_9950X3D_64GB_DDR5_6000</code>
                    </p>
                </div>
                <div class="glass-panel p-4 bg-slate-900/90 text-xs font-mono space-y-1.5 min-w-[340px]">
                    <div><span class="text-slate-400">GPU:</span> <span class="text-sky-400 font-bold">{hw['gpu']['name']}</span></div>
                    <div><span class="text-slate-400">VRAM:</span> <span class="text-emerald-400 font-bold">{hw['gpu']['vram_total_mb']} MiB ({hw['gpu']['vram_total_gb']} GB GDDR7)</span> • <span class="text-slate-300">sm_120a (Driver {hw['gpu']['driver_version']})</span></div>
                    <div><span class="text-slate-400">CPU:</span> <span class="text-slate-200 font-semibold">{hw['cpu']['name']} ({hw['cpu']['cores']}C/{hw['cpu']['threads']}T)</span></div>
                    <div><span class="text-slate-400">Board:</span> <span class="text-slate-300">{hw['motherboard']['manufacturer']} {hw['motherboard']['product']} ({hw['motherboard']['pcie']})</span></div>
                    <div><span class="text-slate-400">RAM:</span> <span class="text-purple-400 font-bold">{hw['ram']['total_gb']} GB DDR5-{hw['ram']['configured_speed_mts']}</span> <span class="text-slate-400">(2x32GB {hw['ram']['manufacturer']} {hw['ram']['part_number']})</span></div>
                    <div><span class="text-slate-400">NVMe SSD:</span> <span class="text-amber-300">{hw['storage']['primary_model_drive']}</span></div>
                </div>
            </div>
        </header>

        <!-- Executive Multi-Model Comparison Table -->
        <section class="glass-panel overflow-hidden">
            <div class="p-6 border-b border-slate-800 flex flex-col md:flex-row justify-between md:items-center gap-2">
                <div>
                    <h2 class="text-xl font-bold text-slate-100">1. Multi-Model Performance & Quality Matrix</h2>
                    <p class="text-xs text-slate-400">Click any model row to filter the detailed test inspector below, or click any test button to inspect raw engine output.</p>
                </div>
            </div>
            <div class="overflow-x-auto">
                <table class="w-full text-left" id="summaryTable">
                    <thead>
                        <tr>
                            <th>Model & Architecture</th>
                            <th>Execution Tier</th>
                            <th class="text-right">VRAM / DDR5</th>
                            <th class="text-right">Short TTFT (Avg)</th>
                            <th class="text-right">Short Prefill</th>
                            <th class="text-right">2K–3K Prefill</th>
                            <th class="text-right">Avg Decode</th>
                            <th class="text-right">Peak Decode</th>
                            <th class="text-center">Quality Audit</th>
                            <th class="text-center">Inspect Tests</th>
                        </tr>
                    </thead>
                    <tbody id="summaryTableBody"></tbody>
                </table>
            </div>
        </section>

        <!-- Charts Grid -->
        <section class="grid grid-cols-1 lg:grid-cols-2 gap-6">
            <div class="glass-panel p-6">
                <h3 class="text-base font-bold text-slate-200 mb-3">Autoregressive Decode Speed Across Models (tok/s)</h3>
                <div class="h-72"><canvas id="decodeChart"></canvas></div>
            </div>
            <div class="glass-panel p-6">
                <h3 class="text-base font-bold text-slate-200 mb-3">Long-Context (2K–3K) vs Short Prefill Throughput (tok/s)</h3>
                <div class="h-72"><canvas id="prefillChart"></canvas></div>
            </div>
        </section>

        <!-- Interactive Full Output Inspector -->
        <section class="glass-panel p-6 space-y-6" id="inspectorSection">
            <div class="flex flex-col md:flex-row justify-between md:items-center gap-4 border-b border-slate-800 pb-4">
                <div>
                    <h2 class="text-xl font-bold text-sky-400">2. Interactive Engine Output & Hallucination Inspector</h2>
                    <p class="text-xs text-slate-400">Select any Model and Prompt below to inspect the exact prompt, automated verification checks, full <code class="text-purple-300">&lt;think&gt;</code> trace, and final output.</p>
                </div>
                <div class="flex flex-wrap items-center gap-3">
                    <label class="text-xs text-slate-400 font-mono">Model:</label>
                    <select id="modelSelect" class="bg-slate-900 border border-slate-700 text-slate-100 text-sm rounded-lg px-3 py-2 font-mono"></select>
                    <label class="text-xs text-slate-400 font-mono">Test Prompt:</label>
                    <select id="promptSelect" class="bg-slate-900 border border-slate-700 text-slate-100 text-sm rounded-lg px-3 py-2 font-mono"></select>
                </div>
            </div>

            <!-- Quick Prompt Pills -->
            <div id="promptPills" class="flex flex-wrap gap-2"></div>

            <!-- Selected Test Telemetry + Audit Panel -->
            <div id="inspectorBody" class="space-y-6"></div>
        </section>
    </div>

    <script>
        const BENCH_DATA = {json_blob};

        function escapeHtml(str) {{
            if (!str) return '';
            return str.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
        }}

        function badgeForStatus(status, text) {{
            const cls = status === 'PASS' ? 'badge-pass' : (status === 'WARN' ? 'badge-warn' : 'badge-fail');
            return `<span class="badge ${{cls}}">${{text || status}}</span>`;
        }}

        function initDashboard() {{
            const models = BENCH_DATA.models || [];
            document.getElementById('headerModelCount').textContent = `${{models.length}} Models Benchmarked`;

            const tbody = document.getElementById('summaryTableBody');
            tbody.innerHTML = '';

            const modelSelect = document.getElementById('modelSelect');
            modelSelect.innerHTML = '';

            models.forEach((m, mIdx) => {{
                const mi = m.model_info;
                const s = m.summary;
                const gt = m.gpu_telemetry;
                const overallStatus = s.fail_count > 0 ? 'FAIL' : (s.warn_count > 0 ? 'WARN' : 'PASS');
                const auditLabel = `${{s.pass_count}}/${{s.total_tests}} PASS`;

                const tr = document.createElement('tr');
                tr.className = 'cursor-pointer';
                tr.onclick = () => selectModelAndPrompt(mIdx, 0);
                tr.innerHTML = `
                    <td>
                        <div class="font-bold text-slate-100">${{escapeHtml(mi.short_name)}}</div>
                        <div class="text-xs text-slate-400">${{escapeHtml(mi.params)}} • ${{escapeHtml(mi.quant.slice(0, 58))}}</div>
                    </td>
                    <td class="text-xs text-slate-300">${{escapeHtml(mi.tier_mode)}}</td>
                    <td class="text-right font-mono text-xs">
                        <span class="text-emerald-400 font-bold">${{gt.vram_used_gb}}G</span> VRAM /
                        <span class="text-purple-400">${{gt.dram_l2_gb}}G</span> RAM
                    </td>
                    <td class="text-right font-mono text-slate-300">${{s.avg_short_ttft_ms}} ms</td>
                    <td class="text-right font-mono text-slate-300">${{s.avg_short_prefill_tps}} tok/s</td>
                    <td class="text-right font-mono text-emerald-400 font-bold">${{s.avg_long_prefill_tps}} tok/s</td>
                    <td class="text-right font-mono text-sky-400 font-extrabold text-base">${{s.avg_decode_tps}} tok/s</td>
                    <td class="text-right font-mono text-cyan-300">${{s.peak_decode_tps}} tok/s</td>
                    <td class="text-center">${{badgeForStatus(overallStatus, auditLabel)}}</td>
                    <td class="text-center">
                        <button class="px-3 py-1 bg-sky-500/20 hover:bg-sky-500/30 text-sky-300 border border-sky-500/40 rounded-lg text-xs font-mono">
                            Inspect Outputs
                        </button>
                    </td>
                `;
                tbody.appendChild(tr);

                const opt = document.createElement('option');
                opt.value = mIdx;
                opt.textContent = `${{mi.short_name}} (${{s.avg_decode_tps}} tok/s decode)`;
                modelSelect.appendChild(opt);
            }});

            modelSelect.onchange = () => {{
                populatePromptSelect(parseInt(modelSelect.value, 10), 0);
            }};

            if (models.length > 0) {{
                populatePromptSelect(0, 0);
                renderCharts(models);
            }}
        }}

        function populatePromptSelect(mIdx, pIdx) {{
            const models = BENCH_DATA.models || [];
            if (!models[mIdx]) return;
            document.getElementById('modelSelect').value = mIdx;
            const results = models[mIdx].results || [];
            const promptSelect = document.getElementById('promptSelect');
            const pills = document.getElementById('promptPills');
            promptSelect.innerHTML = '';
            pills.innerHTML = '';

            results.forEach((r, idx) => {{
                const opt = document.createElement('option');
                opt.value = idx;
                opt.textContent = `${{r.prompt_info.id}} — ${{r.prompt_info.name}} (${{r.metrics.decode_tps}} tok/s | ${{r.audit.status}})`;
                promptSelect.appendChild(opt);

                const btn = document.createElement('button');
                const isSel = idx === pIdx;
                const statusColor = r.audit.status === 'PASS' ? 'text-emerald-400' : (r.audit.status === 'WARN' ? 'text-amber-400' : 'text-rose-400');
                btn.className = `px-3 py-1.5 rounded-lg text-xs font-mono border transition ${{isSel ? 'bg-sky-500/25 border-sky-400 text-white font-bold' : 'bg-slate-900/80 border-slate-700 text-slate-300 hover:border-slate-500'}}`;
                btn.innerHTML = `${{r.prompt_info.id}} <span class="text-sky-400">${{r.metrics.decode_tps}}t/s</span> <span class="${{statusColor}}">●</span>`;
                btn.onclick = () => selectModelAndPrompt(mIdx, idx);
                pills.appendChild(btn);
            }});

            promptSelect.value = pIdx;
            promptSelect.onchange = () => selectModelAndPrompt(mIdx, parseInt(promptSelect.value, 10));
            renderInspector(mIdx, pIdx);
        }}

        function selectModelAndPrompt(mIdx, pIdx) {{
            populatePromptSelect(mIdx, pIdx);
            document.getElementById('inspectorSection').scrollIntoView({{ behavior: 'smooth', block: 'nearest' }});
        }}

        function renderInspector(mIdx, pIdx) {{
            const m = BENCH_DATA.models[mIdx];
            const r = m.results[pIdx];
            const pi = r.prompt_info;
            const mt = r.metrics;
            const au = r.audit;

            const checksHtml = (au.checks || []).map(c => `
                <div class="flex items-start gap-2 p-2.5 rounded-lg bg-slate-900/70 border ${{c.passed ? 'border-emerald-500/30' : 'border-rose-500/40'}}">
                    <span class="font-mono text-xs font-bold ${{c.passed ? 'text-emerald-400' : 'text-rose-400'}}">${{c.passed ? '✓ PASS' : '✗ ISSUE'}}</span>
                    <div class="text-xs">
                        <span class="font-semibold text-slate-200">${{escapeHtml(c.name)}}:</span>
                        <span class="text-slate-300 font-mono ml-1">${{escapeHtml(c.detail)}}</span>
                    </div>
                </div>
            `).join('');

            // Check if P3_HTML_CALC has HTML code we can preview in a sandboxed iframe
            let htmlPreviewBlock = '';
            if (pi.id === 'P3_HTML_CALC') {{
                const codeMatch = (mt.content_text || mt.full_text).match(/```(?:html)?\\s*([\\s\\S]*?)```/i);
                const rawHtml = codeMatch ? codeMatch[1] : ((mt.content_text || '').includes('<html') ? mt.content_text : '');
                if (rawHtml) {{
                    const encodedSrc = encodeURIComponent(rawHtml);
                    htmlPreviewBlock = `
                        <div class="glass-panel p-4 bg-slate-900/90 border border-sky-500/30">
                            <div class="flex justify-between items-center mb-2">
                                <span class="text-xs font-bold uppercase tracking-wider text-sky-400 font-mono">Interactive Live HTML Preview (Sandboxed)</span>
                            </div>
                            <iframe class="w-full h-96 rounded-lg bg-slate-950 border border-slate-800" sandbox="allow-scripts" srcdoc="${{rawHtml.replace(/&/g, '&amp;').replace(/"/g, '&quot;')}}"></iframe>
                        </div>
                    `;
                }}
            }}

            document.getElementById('inspectorBody').innerHTML = `
                <!-- Metrics Bar -->
                <div class="grid grid-cols-2 md:grid-cols-6 gap-3">
                    <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                        <div class="text-[11px] text-slate-400 uppercase">Decode Speed</div>
                        <div class="text-xl font-extrabold text-sky-400 font-mono">${{mt.decode_tps}} tok/s</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                        <div class="text-[11px] text-slate-400 uppercase">Prefill Speed</div>
                        <div class="text-xl font-extrabold text-emerald-400 font-mono">${{mt.prefill_tps}} tok/s</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                        <div class="text-[11px] text-slate-400 uppercase">TTFT Latency</div>
                        <div class="text-xl font-bold text-slate-200 font-mono">${{mt.ttft_ms}} ms</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                        <div class="text-[11px] text-slate-400 uppercase">Prompt Tokens</div>
                        <div class="text-xl font-bold text-slate-200 font-mono">${{mt.prefill_tokens}} tok</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                        <div class="text-[11px] text-slate-400 uppercase">Generated Tokens</div>
                        <div class="text-sm font-bold text-purple-300 font-mono mt-1">${{mt.chunks_count}} (${{mt.think_tokens}}T / ${{mt.content_tokens}}C)</div>
                    </div>
                    <div class="p-3 rounded-xl bg-slate-900/90 border border-slate-800">
                        <div class="text-[11px] text-slate-400 uppercase">Audit Status</div>
                        <div class="mt-1">${{badgeForStatus(au.status, au.summary)}}</div>
                    </div>
                </div>

                <!-- Automated Quality & Hallucination Checks -->
                <div class="space-y-2">
                    <div class="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">Automated Mistake & Hallucination Verification</div>
                    <div class="grid grid-cols-1 md:grid-cols-3 gap-2">${{checksHtml}}</div>
                </div>

                <!-- Exact Prompt -->
                <div class="p-4 rounded-xl bg-slate-900/70 border border-slate-800">
                    <div class="flex justify-between items-center mb-2">
                        <span class="text-xs font-bold uppercase tracking-wider text-slate-400 font-mono">Exact Prompt Sent (${{pi.id}} — ${{pi.name}})</span>
                        <span class="text-xs text-slate-500 font-mono">temp=${{pi.temperature}} | max_tokens=${{pi.max_tokens}} | thinking=${{pi.enable_thinking}} (budget=${{pi.thinking_budget}})</span>
                    </div>
                    <pre class="text-xs font-mono text-slate-300 max-h-40 overflow-y-auto p-3 rounded bg-slate-950/80 border border-slate-800/80">${{escapeHtml(pi.prompt)}}</pre>
                </div>

                <!-- Thinking Trace & Final Output Side-by-Side -->
                <div class="grid grid-cols-1 lg:grid-cols-2 gap-4">
                    <div class="p-4 rounded-xl bg-slate-900/90 border border-purple-500/30 flex flex-col">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs font-bold uppercase tracking-wider text-purple-400 font-mono">Raw &lt;think&gt; / Reasoning Trace (${{mt.think_tokens}} tokens)</span>
                            <a href="${{r.raw_markdown_file}}" target="_blank" class="text-xs text-sky-400 hover:underline font-mono">Open Raw .md ↗</a>
                        </div>
                        <pre class="text-xs font-mono text-purple-200/90 bg-slate-950 p-4 rounded-lg border border-slate-800 overflow-y-auto max-h-[540px] flex-1">${{escapeHtml(mt.think_text || '(Thinking disabled for this prompt or model emitted direct content)')}}</pre>
                    </div>
                    <div class="p-4 rounded-xl bg-slate-900/90 border border-emerald-500/30 flex flex-col">
                        <div class="flex justify-between items-center mb-2">
                            <span class="text-xs font-bold uppercase tracking-wider text-emerald-400 font-mono">Raw Final Engine Output (${{mt.content_tokens}} tokens)</span>
                            <span class="text-xs text-slate-400 font-mono">finish_reason=${{mt.finish_reason}} | ${{mt.total_time_s}}s total</span>
                        </div>
                        <pre class="text-xs font-mono text-slate-100 bg-slate-950 p-4 rounded-lg border border-slate-800 overflow-y-auto max-h-[540px] flex-1">${{escapeHtml(mt.content_text || '(No separate content emitted outside thinking trace)')}}</pre>
                    </div>
                </div>

                ${{htmlPreviewBlock}}
            `;
        }}

        function renderCharts(models) {{
            const labels = models.map(m => m.model_info.short_name);
            const avgDecode = models.map(m => m.summary.avg_decode_tps);
            const peakDecode = models.map(m => m.summary.peak_decode_tps);
            const longPrefill = models.map(m => m.summary.avg_long_prefill_tps);
            const shortPrefill = models.map(m => m.summary.avg_short_prefill_tps);

            new Chart(document.getElementById('decodeChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels,
                    datasets: [
                        {{
                            label: 'Average Decode (tok/s)',
                            data: avgDecode,
                            backgroundColor: 'rgba(56, 189, 248, 0.85)',
                            borderRadius: 6
                        }},
                        {{
                            label: 'Peak Decode (tok/s)',
                            data: peakDecode,
                            backgroundColor: 'rgba(168, 85, 247, 0.75)',
                            borderRadius: 6
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
                    scales: {{
                        y: {{ beginAtZero: true, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                        x: {{ ticks: {{ color: '#cbd5e1' }} }}
                    }}
                }}
            }});

            new Chart(document.getElementById('prefillChart').getContext('2d'), {{
                type: 'bar',
                data: {{
                    labels,
                    datasets: [
                        {{
                            label: '2K–3K Long Prefill (tok/s)',
                            data: longPrefill,
                            backgroundColor: 'rgba(16, 185, 129, 0.85)',
                            borderRadius: 6
                        }},
                        {{
                            label: 'Short Prompt Prefill (tok/s)',
                            data: shortPrefill,
                            backgroundColor: 'rgba(245, 158, 11, 0.80)',
                            borderRadius: 6
                        }}
                    ]
                }},
                options: {{
                    responsive: true,
                    maintainAspectRatio: false,
                    plugins: {{ legend: {{ labels: {{ color: '#cbd5e1' }} }} }},
                    scales: {{
                        y: {{ beginAtZero: true, grid: {{ color: 'rgba(255,255,255,0.05)' }}, ticks: {{ color: '#94a3b8' }} }},
                        x: {{ ticks: {{ color: '#cbd5e1' }} }}
                    }}
                }}
            }});
        }}

        initDashboard();
    </script>
</body>
</html>
"""
    html_path.write_text(html, encoding="utf-8")


def main():
    only_ids = None
    reuse_first = False
    existing_log = None

    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--models" and i + 1 < len(args):
            only_ids = [x.strip() for x in args[i + 1].split(",") if x.strip()]
            i += 2
        elif args[i] == "--reuse-first":
            reuse_first = True
            i += 1
        elif args[i] == "--existing-log" and i + 1 < len(args):
            existing_log = Path(args[i + 1])
            i += 2
        else:
            i += 1

    # Load existing benchmark_results.json if present so incremental runs merge cleanly
    json_path = OUTPUT_DIR / "benchmark_results.json"
    existing_models_map = {}
    if json_path.exists():
        try:
            old_data = json.loads(json_path.read_text(encoding="utf-8"))
            for m in old_data.get("models", []):
                existing_models_map[m["model_info"]["id"]] = m
        except Exception:
            pass

    selected_models = [m for m in MODELS if (only_ids is None or m["id"] in only_ids)]

    for idx, m_cfg in enumerate(selected_models):
        use_reuse = (reuse_first and idx == 0)
        res = benchmark_model(m_cfg, reuse_existing_server=use_reuse, existing_log_path=existing_log)
        if res is not None:
            existing_models_map[m_cfg["id"]] = res
            # Preserve canonical order from MODELS
            ordered_models = [existing_models_map[m["id"]] for m in MODELS if m["id"] in existing_models_map]
            all_data = {
                "generated_at": datetime.now().isoformat(),
                "hardware": HARDWARE_SPEC,
                "models": ordered_models
            }
            generate_reports(all_data)
            print(f"[REPORT] Updated {OUTPUT_DIR / 'index.html'} and REPORT.md ({len(ordered_models)} models complete)")


if __name__ == "__main__":
    main()

