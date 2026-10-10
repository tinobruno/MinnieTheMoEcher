#!/usr/bin/env python3
"""
run_rigorous_benchmark.py — Rigorous, publication-grade benchmark comparing
MinnieTheMoEcher against llama.cpp on NVIDIA RTX 5060 Ti 16GB (Blackwell).

Hardware:
- GPU: NVIDIA GeForce RTX 5060 Ti 16GB (Blackwell, Compute 12.0)
- CPU: AMD Ryzen 7 5700X3D (8C/16T, 96MB L3)
- RAM: 32 GB DDR4-3200 (Dual Channel)
- SSD: Crucial P3 Plus 4TB NVMe SSD
- OS: Windows 11 Pro 64-bit (Build 26300)
- CUDA: 13.1 Update 1, Driver: 610.88
"""

import os
import sys
import time
import json
import subprocess
import urllib.request
import re

MOECHER_EXE = r"e:\dev\MinnieTheMoEcher\moecher.exe"
MOECHER_MANIFEST = r"E:\moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json"

LLAMA_BENCH_EXE = r"e:\dev\llama.cpp\build\bin\Release\llama-bench.exe"
LLAMA_SERVER_EXE = r"e:\dev\llama.cpp\build\bin\Release\llama-server.exe"
GGUF_MODEL = r"E:\moecher\models\gguf_qwen3_8_27b\Qwen3.8-27B-Q3_K_M.gguf"
GGUF_MMPROJ = r"E:\moecher\models\gguf_qwen3_8_27b\mmproj-Qwen3.8-27B-bf16.gguf"
TEST_IMAGE = r"e:\dev\MinnieTheMoEcher\graphics\moecher_logo_b_clear.jpg"

def get_gpu_memory_for_pid(pid):
    """Query Windows Performance Counters for dedicated and non-local GPU memory."""
    try:
        cmd = f"Get-Counter '\\GPU Process Memory(pid_{pid}*)\\*' | Select-Object -ExpandProperty CounterSamples | Select-Object Path, CookedValue"
        res = subprocess.run(["powershell", "-NoProfile", "-Command", cmd], capture_output=True, text=True, timeout=5)
        dedicated = 0
        non_local = 0
        for line in res.stdout.splitlines():
            if "dedicated usage" in line.lower():
                parts = line.strip().split()
                if len(parts) >= 2:
                    dedicated = int(float(parts[-1]))
            elif "non local usage" in line.lower():
                parts = line.strip().split()
                if len(parts) >= 2:
                    non_local = int(float(parts[-1]))
        return dedicated / (1024**3), non_local / (1024**2) # GB, MB
    except Exception:
        return 0.0, 0.0

print("Benchmark script initialized.")
