# MinnieTheMoEcher: Qwen 3.8 / 27B Quantization Guide

This guide details how to quantize **Qwen 3.8 (27B)** and **Qwen 2.5 (32B / 27B)** for **MinnieTheMoEcher** across different GPU VRAM configurations (12 GB, 16 GB, 24 GB+), with and without multimodal vision capabilities.

---

## Table of Contents
1. [Background: Why Was Qwen Slow or Crashing on 16GB GPUs?](#1-background-why-was-qwen-slow-or-crashing-on-16gb-gpus)
2. [Supported GPU Tiers & Recommended Profiles](#2-supported-gpu-tiers--recommended-profiles)
3. [Memory & Precision Architecture Comparison](#3-memory--precision-architecture-comparison)
4. [Quantization Scripts & How to Use Them](#4-quantization-scripts--how-to-use-them)
   - [Tier 1: 16 GB GPUs (RTX 5060 Ti 16GB, 4060 Ti 16GB) — qwen3.8-27B-Vision-13G](#tier-1-16-gb-gpus-rtx-5060-ti-16gb-4060-ti-16gb--qwen38-27b-vision-13g)
   - [Tier 2: 16 GB GPUs (Hybrid 3-bit / 4-bit) — qwen3.8-27B-Vision-14G](#tier-2-16-gb-gpus-hybrid-3-bit--4-bit--qwen38-27b-vision-14g)
   - [Tier 3: 16 GB GPUs (Text-Centric / 32k Context) — qwen3.8-27B-Text-12G](#tier-3-16-gb-gpus-text-centric--32k-context--qwen38-27b-text-12g)
   - [Tier 4: 24 GB+ GPUs (RTX 3090, RTX 4090, A5000) — qwen3.8-27B-Q4](#tier-4-24-gb-gpus-rtx-3090-rtx-4090-a5000--qwen38-27b-q4)
   - [Tier 5: 12 GB GPUs (RTX 3060 12GB, RTX 4070 12GB) — qwen3.8-27B-Compact-10.5G](#tier-5-12-gb-gpus-rtx-3060-12gb-rtx-4070-12gb--qwen38-27b-compact-105g)
5. [Step-by-Step Workflow](#5-step-by-step-workflow)
   - [Step 1: Download / Extract Base Model](#step-1-download--extract-base-model)
   - [Step 2: Quantize to Target Architecture](#step-2-quantize-to-target-architecture)
   - [Step 3: Run Inference](#step-3-run-inference)
6. [VRAM Budgeting & KV Cache Sizing](#6-vram-budgeting--kv-cache-sizing)
7. [Troubleshooting & Common Pitfalls](#7-troubleshooting--common-pitfalls)

---

## 1. Background: Why Was Qwen Slow or Crashing on 16GB GPUs?

Qwen 3.8 27B contains **27.32 billion parameters**:
- **MLP layers (`gate_proj`, `up_proj`, `down_proj`)**: 17.11B parameters.
- **Attention layers (`linear_attn`, `self_attn`)**: 7.21B parameters.
- **Vocabulary Projections (`embed_tokens` + `lm_head`)**: 2.54B parameters (vocab size: 248,320; hidden size: 5,120).
- **Vision Tower (`model.visual.*`)**: ~0.45B parameters.

### The Naive INT4 Issue:
In early conversion tools (`scripts/quantize_qwen.py`), only the internal projection layers were quantized to INT4. `embed_tokens` (2.54 GB), `lm_head` (2.54 GB), and `model.visual.*` (0.92 GB) were left in uncompressed BF16.
- **Resulting File Size**: **20.06 GB**.
- **On a 24 GB GPU (RTX 3090 / 4090)**: Fits comfortably, running at **40–50+ tok/s**.
- **On a 16 GB GPU (RTX 5060 Ti / 4060 Ti)**: 20 GB exceeds the 16,384 MB physical VRAM limit. CUDA is forced to spill ~4–5 GB into system RAM over PCIe. Because PCIe bandwidth (~16–32 GB/s) is 20–50x slower than VRAM bandwidth (448–1008 GB/s), generation speed collapsed to single digits, or the server crashed on startup.

---

## 2. Supported GPU Tiers & Recommended Profiles

| GPU VRAM | Recommended Profile | Model Footprint | Free VRAM on GPU | Context Limit | Primary Script |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **16 GB** (Windows / Linux) | **`qwen3.8-27B-Vision-13G`** | **~13.1 – 13.5 GB** | **~2.5 – 2.9 GB** | 8k – 16k | `scripts/quantize_qwen_vision.py` |
| **16 GB** (Linux Headless) | **`qwen3.8-27B-Vision-14G`** | **~14.0 GB** | **~2.0 GB** | 4k – 8k | `scripts/quantize_qwen_vision.py --mlp-scheme hybrid` |
| **16 GB** (Text-Centric) | **`qwen3.8-27B-Text-12G`** | **~12.2 GB** | **~3.8 GB** | 32k+ | `scripts/quantize_qwen_vision.py --visual-mode int4` |
| **24 GB+** (3090, 4090) | **`qwen3.8-27B-Q4`** | **~20.0 GB** | **~4.0 GB** | 32k+ | `scripts/quantize_qwen.py` |
| **12 GB** (3060, 4070) | **`qwen3.8-27B-Compact-10.5G`** | **~10.8 GB** | **~1.2 GB** | 4k | `scripts/quantize_qwen_vision.py --visual-mode int4` |

---

## 3. Memory & Precision Architecture Comparison

The table below details how individual model components are treated across profiles:

| Component | Layer Identifier | BF16 Size | 24GB Profile (`Q4`) | 16GB Profile (`Vision-13G`) | 12GB Profile (`Compact`) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Visual Tower** | `model.visual.*` | ~0.92 GB | BF16 (0.92 GB) | **BF16 Raw (0.92 GB)** *(100% OCR & Vision)* | INT4 (0.23 GB) |
| **Embeddings** | `embed_tokens.weight` | 2.54 GB | BF16 (2.54 GB) | **INT4 Block-32 (0.72 GB)** | INT4 Block-32 (0.72 GB) |
| **LM Head** | `lm_head.weight` | 2.54 GB | BF16 (2.54 GB) | **INT4 Block-32 (0.72 GB)** | INT4 Block-32 (0.72 GB) |
| **Attention** | `linear_attn.*`, `self_attn.*` | 6.34 GB | INT4 (3.17 GB) | **INT4 Block-32 (3.17 GB)** | INT4 Block-32 (3.17 GB) |
| **MLP Projections** | `gate_proj`, `up_proj`, `down_proj` | 10.16 GB | INT4 (5.08 GB) | **INT3 Block-32 (3.74 GB)** *(3.50 bpw)* | INT3 Block-32 (3.74 GB) |
| **Norms & Convs** | `*_norm`, `conv1d` | ~0.10 GB | BF16 (0.10 GB) | **BF16 (0.10 GB)** | BF16 (0.10 GB) |
| **Total Size** | | **22.60 GB** | **20.06 GB** | **~13.1 – 13.5 GB** | **~10.8 GB** |

### Why INT3 Block-32 for MLP?
Feed-forward MLP layers account for 62.6% of the parameters (17.11B). By applying symmetric 3-bit quantization with block size 32:
- 8 weights pack into 3 bytes (24 bits).
- Each block of 32 weights uses 12 packed bytes + 2 bytes for a `bfloat16` scale factor.
- Effective bitwidth: **3.50 bits per weight** (4.57x compression vs BF16).
- Accuracy: Retains higher perplexity stability than naive INT3 by preserving block-level dynamic dynamic ranges.

---

## 4. Quantization Scripts & How to Use Them

### Tier 1: 16 GB GPUs (RTX 5060 Ti 16GB, 4060 Ti 16GB) — `qwen3.8-27B-Vision-13G`
> **Target**: Fits fully in VRAM (~13.1 GB), leaving ~2.5 to 2.9 GB free for KV cache and OS. Preserves 100% full-resolution vision & OCR.

#### Option A: Converting from an existing local Qwen checkpoint (`models/qwen3_8_27b_q4`)
```bash
python3 scripts/quantize_qwen_vision.py \
    --manifest-in models/qwen3_8_27b_q4/moecher_manifest.json \
    --output-dir models/qwen3.8-27B-Vision-13G
```
*(Note: `scripts/quantize_qwen_vision.py` automatically detects `attention_dense_layers_q4.bin` in the manifest's folder).*

#### Option B: Converting directly from raw Hugging Face safetensors
```bash
python3 scripts/quantize_qwen_vision.py \
    --input-dir /path/to/raw_qwen_hf \
    --output-dir models/qwen3.8-27B-Vision-13G
```

---

### Tier 2: 16 GB GPUs (Hybrid 3-bit / 4-bit) — `qwen3.8-27B-Vision-14G`
> **Target**: ~14.0 GB footprint. Keeps `down_proj` in 4-bit while quantizing `gate_proj` and `up_proj` in 3-bit. Leaves ~2.0 GB free on 16GB cards.

```bash
python3 scripts/quantize_qwen_vision.py \
    --manifest-in models/qwen3_8_27b_q4/moecher_manifest.json \
    --mlp-scheme hybrid \
    --output-dir models/qwen3.8-27B-Vision-14G
```

---

### Tier 3: 16 GB GPUs (Text-Centric / 32k Context) — `qwen3.8-27B-Text-12G`
> **Target**: ~12.2 GB footprint. Quantizes the visual tower to INT4, freeing up an extra ~700 MB of VRAM for larger KV cache context windows (up to 32k tokens).

```bash
python3 scripts/quantize_qwen_vision.py \
    --manifest-in models/qwen3_8_27b_q4/moecher_manifest.json \
    --visual-mode int4 \
    --output-dir models/qwen3.8-27B-Text-12G
```

---

### Tier 4: 24 GB+ GPUs (RTX 3090, RTX 4090, A5000) — `qwen3.8-27B-Q4`
> **Target**: ~20.0 GB footprint. Keeps `embed_tokens`, `lm_head`, and `model.visual` in uncompressed BF16 while quantizing dense projections to INT4 block-32.

```bash
python3 scripts/quantize_qwen.py \
    --input-dir models/qwen3_8_27b \
    --output-dir models/qwen3_8_27b_q4 \
    --block-size 32
```

---

### Tier 5: 12 GB GPUs (RTX 3060 12GB, RTX 4070 12GB) — `qwen3.8-27B-Compact-10.5G`
> **Target**: ~10.8 GB footprint. Uses INT3 MLP + INT4 Vision + INT4 Attention + INT4 Embeddings to run within 12 GB VRAM.

```bash
python3 scripts/quantize_qwen_vision.py \
    --manifest-in models/qwen3_8_27b_q4/moecher_manifest.json \
    --visual-mode int4 \
    --mlp-bits 3 \
    --mlp-scheme all-3bit \
    --output-dir models/qwen3.8-27B-Compact-10.5G
```

---

## 5. Step-by-Step Workflow

### Step 1: Download / Extract Base Model
If you do not have a local binary yet, prepare the base Qwen model from Hugging Face:
```bash
python3 scripts/prepare_qwen.py \
    --model-id Qwen/Qwen2.5-Coder-32B-Instruct \
    --output-dir models/qwen3_8_27b
```
This creates `models/qwen3_8_27b/moecher_manifest_qwen.json` and extracts the safetensors.

### Step 2: Quantize to Target Architecture
For a 16 GB GPU (e.g. RTX 5060 Ti 16GB):
```bash
python3 scripts/quantize_qwen_vision.py \
    --manifest-in models/qwen3_8_27b/moecher_manifest_qwen.json \
    --output-dir models/qwen3.8-27B-Vision-13G
```

### Step 3: Run Inference
Launch **MinnieTheMoEcher** using the generated manifest:
```bash
./build/moecher \
    --manifest models/qwen3.8-27B-Vision-13G/moecher_manifest.json \
    --port 8001 \
    --quiet
```

---

## 6. VRAM Budgeting & KV Cache Sizing

On a 16 GB card (16,384 MB physical memory):
```
Total Physical VRAM:  16,384 MB
Operating System:     ~800 - 1,100 MB (Windows 11 Desktop) / ~200 MB (Linux Headless)
Weights (Vision-13G): ~13,100 MB
Available for Cache:  ~2,184 - 2,484 MB
```

### KV Cache Memory by Context Length:
Qwen 3.8 27B uses Grouped-Query Attention (GQA) with `num_key_value_heads = 4`, `head_dim = 256`, and `num_hidden_layers = 64`.
Per-token KV cache footprint:
$$\text{Bytes per token} = 2 \times 64 \times 4 \times 256 \times 2 \text{ bytes (BF16)} = 262,144 \text{ bytes (0.25 MB/token)}$$

| Context Length | KV Cache Size | Fits on 16GB with `Vision-13G`? |
| :--- | :--- | :--- |
| **2,048 tokens** | **512 MB** | Yes (Plenty of headroom) |
| **4,096 tokens** | **1,024 MB** | Yes (Comfortable) |
| **8,192 tokens** | **2,048 MB** | Yes (Optimal) |
| **16,384 tokens** | **4,096 MB** | Requires Linux headless or `Text-12G` profile |

---

## 7. Troubleshooting & Common Pitfalls

### 1. Multi-Line Command Error (`--dense-bin-in: command not found`)
If you copy and paste a multi-line terminal command and the shell splits on an unescaped newline:
```bash
# INCORRECT (splits into two separate shell commands):
python3 scripts/quantize_qwen_vision.py --manifest-in models/qwen3_8_27b_q4/moecher_manifest.json
    --dense-bin-in models/qwen3_8_27b_q4/attention_dense_layers_q4.bin

# CORRECT (single line):
python3 scripts/quantize_qwen_vision.py --manifest-in models/qwen3_8_27b_q4/moecher_manifest.json --output-dir models/qwen3.8-27B-Vision-13G
```
*Note: `scripts/quantize_qwen_vision.py` automatically resolves the companion binary file when `--manifest-in` is provided.*

### 2. Slow Generation (20–25 tok/s instead of 40–80 tok/s)
- **Check Model Size**: Verify that `attention_dense_layers.bin` is **$\le 13.5$ GB**. If it is 20 GB, you are running the uncompressed 24GB profile on a 16GB card, causing PCIe swapping.
- **Check Prefill Mode**: Ensure batched prefill is active on startup (`[Prefill] Batched prefill enabled`).
- **Check CUDA Graph**: Verify that CUDA Graph warmup succeeded (`[INFO] CUDA Graph instantiated successfully!`).

### 3. Out of Memory (CUDA OOM) on Windows
Windows Desktop Window Manager (DWM) uses 800 MB – 1.2 GB of VRAM. If OOM occurs:
1. Use `--max-seq-len 8192` to limit pre-allocated KV cache.
2. Or use `--visual-mode int4` to free up ~700 MB.
