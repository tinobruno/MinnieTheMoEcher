# MinnieTheMoEcher: Quantization & Manifest Guide

This document provides a comprehensive, step-by-step guide to calibrating, quantizing, packaging, and deploying DeepSeek-V4 MoE expert binaries and manifests for **MinnieTheMoEcher**.

---

## Table of Contents
1. [Overview & Supported Quantization Formats](#1-overview--supported-quantization-formats)
2. [Binary Layouts & Memory Specifications](#2-binary-layouts--memory-specifications)
3. [Step 1: Calibration & Importance Matrix (`.dat`) Generation](#3-step-1-calibration--importance-matrix-dat-generation)
4. [Step 2: Expert Quantization (`.bin`)](#4-step-2-expert-quantization-bin)
5. [Step 3: Manifest Configuration (`.json`)](#5-step-3-manifest-configuration-json)
6. [Alternative: Direct GGUF Extraction](#6-alternative-direct-gguf-extraction)
7. [Step 4: Running Inference & Multi-Tier Caching](#7-step-4-running-inference--multi-tier-caching)
8. [Verification & Numerical Parity Testing](#8-verification--numerical-parity-testing)

---

## 1. Overview & Supported Quantization Formats

MinnieTheMoEcher runs DeepSeek-V4 MoE (43 transformer layers $\times$ 256 routed experts = 11,008 total active routed experts, plus 3 MTP layers = 11,776 total experts).

| Format | Gate (`w1`) | Up (`w3`) | Down (`w2`) | Size per Expert | Total (11,008 Exps) | Total (11,776 Exps) |
|---|---|---|---|---|---|---|
| **`fp4`** (Baseline) | FP4 E2M1 (4-bit) | FP4 E2M1 (4-bit) | FP4 E2M1 (4-bit) | 13.36 MB | 147.1 GiB | 157.4 GiB |
| **`iq2_xxs`** (Optimal) | IQ2_XXS (2.06 bpw) | IQ2_XXS (2.06 bpw) | Q2_K (2.56 bpw) | 6.75 MB (7,077,888 B) | **72.56 GiB** | **77.62 GiB** |
| **`int2`** (Asymmetric) | INT2 (2-bit uniform) | INT2 (2-bit uniform) | INT2 (2-bit uniform) | 6.88 MB (7,208,960 B) | **75.68 GiB** | **80.97 GiB** |

---

## 2. Binary Layouts & Memory Specifications

Each expert contains three weight matrices:
- **`w1` (Gate Projection)**: $[2048, 4096]$ logical weights
- **`w3` (Up Projection)**: $[2048, 4096]$ logical weights
- **`w2` (Down Projection)**: $[4096, 2048]$ logical weights

### Layout in `moe_experts_iq2.bin`
Every expert block is packed sequentially into exactly **7,077,888 bytes** (page-aligned, 1,728 pages $\times$ 4096 bytes):

```
┌────────────────────────────────────────┬────────────────────────────────────────┬────────────────────────────────────────┐
│ w1.weight (IQ2_XXS)                    │ w3.weight (IQ2_XXS)                    │ w2.weight (Q2_K)                       │
│ 2,162,688 Bytes (32,768 blocks x 66B)  │ 2,162,688 Bytes (32,768 blocks x 66B)  │ 2,752,512 Bytes (32,768 blocks x 84B)  │
│ Offset: 0                              │ Offset: 2,162,688                      │ Offset: 4,325,376                      │
└────────────────────────────────────────┴────────────────────────────────────────┴────────────────────────────────────────┘
Total per expert block: 7,077,888 Bytes
```

- **`block_iq2_xxs` structure (66 bytes per 256 weights)**:
  - `uint16_t d`: 16-bit float super-scale ($E_8$ lattice amplitude scale).
  - `uint32_t qs[16]`: 16 $\times$ 32-bit words containing 8 sub-blocks:
    - 8-bit grid index into the 256-entry Gosset lattice $E_8$ codebook (`GRID_TENSOR`).
    - 7-bit parity-checked sign index (`KSIGNS_IQ2XS`).
    - 4-bit sub-block scale delta $\in [0..15]$ ($db = d \cdot (0.5 + \text{delta}) \cdot 0.25$).

- **`block_q2_K` structure (84 bytes per 256 weights)**:
  - `uint8_t scales[16]`: 16 sub-block 4-bit scale (`& 0x0F`) and 4-bit min (`>> 4`) factors.
  - `uint8_t qs[64]`: 256 2-bit quantized values (4 values packed per byte).
  - `uint16_t d`: 16-bit float super-scale.
  - `uint16_t dmin`: 16-bit float minimum baseline scale.

---

## 3. Step 1: Calibration & Importance Matrix (`.dat`) Generation

Activation importance matrices preserve the numerical sensitivity of high-variance channels during low-bit quantization.

### Option A: Interactive Calibration Menu
```bash
python3 scripts/build_imatrix.py
```
1. Select dataset `[4]` (Balanced Multi-Domain: WikiText-2 + GSM8K + Systems Code).
2. Enter token calibration budget: `100000` (recommended for full 11,776-expert coverage).
3. The engine will stream the tokens through all GPU layers and write `imatrix/balanced_imatrix.dat` (431 MB binary file).

### Option B: Command-Line One-Liner
```bash
python3 scripts/build_imatrix.py \
  --dataset balanced-multidomain \
  --prefix balanced \
  --max-tokens 100000
```

### Option C: Direct Bare-Metal C++/CUDA Engine Streaming
```bash
./build/moecher \
  --manifest moecher_manifest.json \
  --imatrix-dataset data/calibration/balanced_multidomain.txt \
  --imatrix-out imatrix/balanced_imatrix.dat \
  --imatrix-max-tokens 100000
```

---

## 4. Step 2: Expert Quantization (`.bin`)

To quantize the expert weights with full parity-aware sign search and least-squares scaling:

### Clean IQ2_XXS + Q2_K Quantization (Unweighted / Pure Lattice)
```bash
python3 scripts/quantize_experts_iq2.py \
  --manifest moecher_manifest.json \
  --output-bin moe_experts_iq2.bin \
  --output-manifest moecher_manifest_iq2.json \
  --batch-size 32
```

### Calibrated IQ2_XXS + Q2_K Quantization (With Importance Matrix)
```bash
python3 scripts/quantize_experts_iq2.py \
  --manifest moecher_manifest.json \
  --imatrix imatrix/balanced_imatrix.dat \
  --output-bin moe_experts_balanced_iq2.bin \
  --output-manifest moecher_manifest_balanced_iq2.json \
  --batch-size 32
```

### Fast Self-Test
Verify quantization kernels without writing the full binary:
```bash
python3 scripts/quantize_experts_iq2.py --manifest moecher_manifest.json --test
```

---

## 5. Step 3: Manifest Configuration (`.json`)

The manifest links the model architecture, dense weight tables, and expert storage offsets.

### Manifest Structure
```json
{
  "model_config": {
    "vocab_size": 129280,
    "hidden_size": 4096,
    "num_hidden_layers": 43,
    "num_attention_heads": 64,
    "num_key_value_heads": 1,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1024,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "moe_intermediate_size": 2048,
    "n_routed_experts": 256,
    "num_experts_per_tok": 6,
    "n_shared_experts": 1,
    "n_hash_layers": 3,
    "rms_norm_eps": 1e-06,
    "rope_theta": 10000,
    "rope_factor": 16,
    "original_seq_len": 65536,
    "rope_beta_fast": 32,
    "rope_beta_slow": 1,
    "sliding_window": 128,
    "scoring_func": "sqrtsoftplus",
    "routed_scaling_factor": 1.5,
    "swiglu_limit": 10.0,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-06,
    "bos_token_id": 0,
    "eos_token_id": 1,
    "compress_ratios": [0, 0, 4, 128, 4, 128, ...],
    "compress_rope_theta": 160000,
    "index_n_heads": 64,
    "index_head_dim": 128,
    "index_topk": 512,
    "expert_dtype": "iq2_xxs",
    "torch_dtype": "bfloat16"
  },
  "tokenizer": {
    "bos_token_id": 0,
    "eos_token_id": 1,
    "user_token": "<｜User｜>",
    "assistant_token": "<｜Assistant｜>",
    "tokenizer_json": "/path/to/tokenizer.json"
  },
  "dense_bin": "/path/to/attention_dense_layers.bin",
  "expert_bin": "/path/to/moe_experts_iq2.bin",
  "dense_tensors": {
    "embed.weight": {
      "offset": 0,
      "nbytes": 1059061760,
      "dtype": "BF16",
      "shape": [129280, 4096]
    },
    ...
  },
  "expert_layout": {
    "block_size": 7077888,
    "n_layers": 46,
    "n_experts": 256,
    "parts": {
      "w1.weight": {
        "offset_in_block": 0,
        "nbytes": 2162688,
        "dtype": "iq2_xxs",
        "shape": [2048, 4096]
      },
      "w3.weight": {
        "offset_in_block": 2162688,
        "nbytes": 2162688,
        "dtype": "iq2_xxs",
        "shape": [2048, 4096]
      },
      "w2.weight": {
        "offset_in_block": 4325376,
        "nbytes": 2752512,
        "dtype": "q2_k",
        "shape": [4096, 2048]
      }
    },
    "part_order": ["w1.weight", "w3.weight", "w2.weight"]
  }
}
```

---

## 6. Alternative: Direct GGUF Extraction

If you already have a quantized DeepSeek-V4 GGUF file (`IQ2_XXS` / `Q2_K`), you can directly extract the expert binary without re-quantizing:

```bash
python3 scripts/extract_gguf_experts.py \
  --gguf path/to/model.gguf \
  --manifest-in moecher_manifest.json \
  --output-bin moe_experts_iq2.bin \
  --output-manifest moecher_manifest_iq2.json
```

---

## 7. Step 4: Running Inference & Multi-Tier Caching

### Start Server with Dual-Tier Cache Allocation
```bash
./build/moecher \
  --manifest moecher_manifest_iq2.json \
  --max-vram 88 \
  --dram-cache-gb 45 \
  --quiet
```

### CLI Parameters:
- `--manifest <path>`: Path to manifest JSON.
- `--max-vram <GB>`: Total VRAM allocation budget (e.g. `88` for 96 GB card). Model dense weights (~13.4 GB) and KV cache are loaded first; remaining VRAM is used for L1 expert cache.
- `--dram-cache-gb <GB>`: Host pinned memory buffer (L2 cache) for offloaded experts (zero-copy PCIe DMA transfer).
- `--shared-expert-dtype <fp4|int8|bf16>`: Sets shared expert precision.
- `--port <N>`: HTTP API port (default: `8001`).
- `--quiet`: Suppresses verbose token streaming in server console.

---

## 8. Verification & Numerical Parity Testing

### 1. Real Expert Numerical Parity Test
Verify CUDA dequantization against PyTorch CPU reference across all matrix dimensions:
```bash
./build/test_real_expert_parity
```

### 2. SwiGLU Fused Kernel Output Validation
Ensure fused `IQ2_XXS` Gate + Up SwiGLU kernel matches reference dequantized outputs:
```bash
./build/test_swiglu_compare
```

### 3. End-to-End Chat Test
```bash
python3 tests/test_pi.py
```
Or interact via CLI chat:
```bash
python3 chat.py --prompt "Explain the Gosset E8 lattice in 2 sentences."
```
