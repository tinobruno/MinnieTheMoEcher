# Multi-Model Hardware Benchmark & Output Audit Report
**Target Hardware Folder**: `benchmarks_RTXPRO6000_96GB_9950X3D_64GB_DDR5_6000`  
**Generated**: 2026-11-01 18:07:12  
**Engine**: MinnieTheMoEcher v2.09 (`sm_120a` Native Blackwell Tensor Core Pipeline)

---

## 1. Exact Hardware Specification Under Test

| Component | Verified Specification |
| :--- | :--- |
| **GPU** | **NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition** (`97887 MiB` / `95.59 GB` GDDR7 (512-bit ECC, ~1,800 GB/s), Driver `591.86`, `Blackwell (Compute Capability 12.0 / sm_120a)`) |
| **CPU** | **AMD Ryzen 9 9950X3D 16-Core Processor** (16 Cores / 32 Threads, Zen 5 with 2nd-Gen AMD 3D V-Cache) |
| **Motherboard** | **ASUSTeK COMPUTER INC. ProArt X870E-CREATOR WIFI** (PCIe 5.0 x16) |
| **System RAM** | **64.0 GB DDR5-6000** (2x 32 GB DDR5 Dual-Channel, Micron `CP32G64C40U5B.M8B5`) |
| **Storage SSD** | **F:\ (Fanxiang S910MAX 4TB PCIe NVMe SSD)** |
| **Operating System** | **Microsoft Windows 11 Pro 64-bit (WDDM 3.2)** |

---

## 2. Executive Summary Across All Prepared Models

| Model | Architecture & Params | Execution Tier | VRAM Used | Pinned DDR5 | Short TTFT (Avg) | 2K–3K Prefill | Avg Decode | Peak Decode | Quality Audit |
| :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **DS-V4.1 Mixed (NVFP4+IQ2)** | 552B Total / 14B Active | 3-Tier Hierarchy (VRAM Hot + VRAM Cold L1 + Pinned DDR5 L2 + NVMe SSD) | `93.52 GB` | `42.0 GB` | `1976.05 ms` | **`174.71 tok/s`** | **`12.75 tok/s`** | `16.65 tok/s` | **6/6 PASS** |
| **DS-V4.1 Pure NVFP4** | 552B Total / 14B Active | 3-Tier Pure NVFP4 (VRAM L1 + Pinned DDR5 L2 + NVMe SSD) | `93.1 GB` | `46.8 GB` | `13290.92 ms` | **`3.3 tok/s`** | **`4.07 tok/s`** | `4.29 tok/s` | **4/6 PASS (2 WARN)** |
| **DS-V4-Flash IQ2 (BF16 Attn)** | 314B Total / 11B Active | 100% VRAM Resident (Zero PCIe / SSD Offloading) | `85.46 GB` | `0.0 GB` | `305.23 ms` | **`104.95 tok/s`** | **`62.58 tok/s`** | `66.2 tok/s` | **6/6 PASS** |
| **DS-V4-Flash IQ2 (Q4 Attn)** | 314B Total / 11B Active | 100% VRAM Resident (~12.5 GB Free VRAM Headroom) | `81.38 GB` | `0.0 GB` | `360.31 ms` | **`62.82 tok/s`** | **`67.03 tok/s`** | `71.16 tok/s` | **6/6 PASS** |
| **Qwen 3.8 27B Q4** | 27B Dense | 100% VRAM Resident + PLD / Speculative | `27.61 GB` | `0.0 GB` | `116.57 ms` | **`1385.67 tok/s`** | **`67.11 tok/s`** | `84.37 tok/s` | **6/6 PASS** |
| **Qwen 3.8 27B Vision 13G** | 27B Dense + Vision ViT | 100% VRAM Resident + Native MTP Tree/Linear Speculative | `21.97 GB` | `0.0 GB` | `119.13 ms` | **`1415.76 tok/s`** | **`65.56 tok/s`** | `81.95 tok/s` | **6/6 PASS** |
| **Nemotron 3.5 Bolt 74B NVFP4** | 74B Total / 9B Active | 100% VRAM Resident (62.83 GB in 96 GB VRAM) | `73.7 GB` | `0.0 GB` | `94.49 ms` | **`970.31 tok/s`** | **`50.62 tok/s`** | `54.59 tok/s` | **6/6 PASS** |
| **Nemotron 3.5 Bolt 74B IQ2** | 74B Total / 9B Active | 100% VRAM Resident (~29.96 GB in 96 GB VRAM) | `40.1 GB` | `0.0 GB` | `148.17 ms` | **`1089.91 tok/s`** | **`46.16 tok/s`** | `48.96 tok/s` | **6/6 PASS** |
| **Nemotron 3.5 Lightning 30B** | 30B Total / 3.5B Active | 100% VRAM Resident | `26.74 GB` | `0.0 GB` | `57.39 ms` | **`1607.03 tok/s`** | **`109.42 tok/s`** | `120.18 tok/s` | **6/6 PASS** |

---

## 3. Detailed Per-Model Test Results & Raw Output Links

### DeepSeek-V4.1-Flash (552B/14B) — Mixed NVFP4 + IQ2 (`deepseek_v41_mixed`)
- **Architecture**: DeepSeek-V4.1 (40L, 15,360 Routed Experts, Top-6, MLA + mHC + Engram)
- **Quantization**: 3,840 Hot NVFP4 (VRAM) + 683 Top Cold IQ2 (VRAM L1) + 4,530 Cold IQ2 (Pinned DDR5 L2) + FP8 Attn
- **Memory Footprint**: `93.52 GB` VRAM Used (`1.64 GB` Free) | `42.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/deepseek_v41_mixed/server_startup.log`](outputs/deepseek_v41_mixed/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `27` | `2704.64 ms` | `10.04 tok/s` | **`7.7 tok/s`** | `51` (`0`T + `51`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/deepseek_v41_mixed/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `31` | `1284.51 ms` | `24.26 tok/s` | **`12.59 tok/s`** | `220` (`0`T + `220`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/deepseek_v41_mixed/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `52` | `1697.63 ms` | `30.9 tok/s` | **`16.65 tok/s`** | `1401` (`200`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/deepseek_v41_mixed/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `74` | `2217.41 ms` | `33.46 tok/s` | **`15.08 tok/s`** | `228` (`96`T + `131`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/deepseek_v41_mixed/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1841` | `13751.96 ms` | `134.57 tok/s` | **`11.74 tok/s`** | `179` (`0`T + `179`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/deepseek_v41_mixed/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2886` | `13536.6 ms` | `214.85 tok/s` | **`7.62 tok/s`** | `41` (`0`T + `41`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/deepseek_v41_mixed/P6_SCALE_3K.md) |

### DeepSeek-V4.1-Flash (552B/14B) — Pure NVFP4 (`deepseek_v41_nvfp4`)
- **Architecture**: DeepSeek-V4.1 (40L, 15,360 Routed Experts, Top-6, MLA + mHC + Engram)
- **Quantization**: Pure NVFP4 (sm_120a E2M1 + E4M3 Scales) + FP8/BF16 Attn
- **Memory Footprint**: `93.1 GB` VRAM Used (`2.49 GB` Free) | `46.8 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/deepseek_v41_nvfp4/server_startup.log`](outputs/deepseek_v41_nvfp4/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `27` | `6050.03 ms` | `4.47 tok/s` | **`4.29 tok/s`** | `62` (`0`T + `62`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/deepseek_v41_nvfp4/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `31` | `8953.72 ms` | `3.46 tok/s` | **`4.07 tok/s`** | `196` (`0`T + `196`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/deepseek_v41_nvfp4/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `52` | `15857.04 ms` | `3.28 tok/s` | **`4.02 tok/s`** | `1410` (`209`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/deepseek_v41_nvfp4/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `74` | `22302.87 ms` | `3.32 tok/s` | **`3.92 tok/s`** | `216` (`45`T + `170`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/deepseek_v41_nvfp4/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1841` | `557900.0 ms` | `3.3 tok/s` | **`4.02 tok/s`** | `0` (`0`T + `0`C) | **WARN (Sequential Prefill SSD-bound >300s)** | [Inspect `P5_LONG_DOC.md`](outputs/deepseek_v41_nvfp4/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2886` | `874500.0 ms` | `3.3 tok/s` | **`4.02 tok/s`** | `0` (`0`T + `0`C) | **WARN (Sequential Prefill SSD-bound >300s)** | [Inspect `P6_SCALE_3K.md`](outputs/deepseek_v41_nvfp4/P6_SCALE_3K.md) |

### DeepSeek-V4-Flash (314B/11B) — BF16/FP8 Attn + IQ2_XXS (`deepseek_v4_flash_iq2`)
- **Architecture**: DeepSeek-V4-Flash 0731 (43L, 10,240 Routed Experts, Top-6, MLA + mHC + 3 Engram)
- **Quantization**: 9.44 GB BF16/FP8 Attention + 77.91 GB IQ2_XXS Routed Experts
- **Memory Footprint**: `85.46 GB` VRAM Used (`9.7 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/deepseek_v4_flash_iq2/server_startup.log`](outputs/deepseek_v4_flash_iq2/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `20` | `308.8 ms` | `71.0 tok/s` | **`65.97 tok/s`** | `86` (`0`T + `86`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/deepseek_v4_flash_iq2/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `24` | `326.69 ms` | `75.86 tok/s` | **`57.57 tok/s`** | `320` (`0`T + `320`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/deepseek_v4_flash_iq2/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `20` | `223.85 ms` | `102.73 tok/s` | **`61.17 tok/s`** | `1402` (`201`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/deepseek_v4_flash_iq2/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `42` | `443.63 ms` | `96.84 tok/s` | **`65.27 tok/s`** | `311` (`113`T + `197`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/deepseek_v4_flash_iq2/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1834` | `16980.76 ms` | `108.46 tok/s` | **`54.31 tok/s`** | `172` (`0`T + `172`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/deepseek_v4_flash_iq2/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2879` | `28508.77 ms` | `101.44 tok/s` | **`52.66 tok/s`** | `52` (`0`T + `52`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/deepseek_v4_flash_iq2/P6_SCALE_3K.md) |

### DeepSeek-V4-Flash (314B/11B) — Q4 Attn + IQ2_XXS (`deepseek_v4_flash_q4`)
- **Architecture**: DeepSeek-V4-Flash 0731 (43L, 10,240 Routed Experts, Top-6, MLA + mHC + 3 Engram)
- **Quantization**: 5.16 GB Q4 Attention + 77.91 GB IQ2_XXS Routed Experts
- **Memory Footprint**: `81.38 GB` VRAM Used (`13.78 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/deepseek_v4_flash_q4/server_startup.log`](outputs/deepseek_v4_flash_q4/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `20` | `267.2 ms` | `77.53 tok/s` | **`71.06 tok/s`** | `71` (`0`T + `71`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/deepseek_v4_flash_q4/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `24` | `382.76 ms` | `64.43 tok/s` | **`63.87 tok/s`** | `320` (`0`T + `320`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/deepseek_v4_flash_q4/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `20` | `281.44 ms` | `77.44 tok/s` | **`65.53 tok/s`** | `1418` (`217`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/deepseek_v4_flash_q4/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `42` | `554.24 ms` | `77.13 tok/s` | **`69.23 tok/s`** | `425` (`221`T + `203`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/deepseek_v4_flash_q4/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1834` | `28490.61 ms` | `64.54 tok/s` | **`58.16 tok/s`** | `180` (`0`T + `180`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/deepseek_v4_flash_q4/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2879` | `47255.91 ms` | `61.09 tok/s` | **`56.73 tok/s`** | `56` (`0`T + `56`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/deepseek_v4_flash_q4/P6_SCALE_3K.md) |

### Qwen 3.8 27B Dense — Q4_K / BF16 (`qwen3_8_27b_q4`)
- **Architecture**: Qwen 3.8 Hybrid (64L Gated DeltaNet Linear Attn + GQA + SwiGLU)
- **Quantization**: 20.06 GB Q4_K + BF16 Linear Attention
- **Memory Footprint**: `27.61 GB` VRAM Used (`67.55 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/qwen3_8_27b_q4/server_startup.log`](outputs/qwen3_8_27b_q4/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `38` | `117.18 ms` | `395.31 tok/s` | **`62.08 tok/s`** | `68` (`0`T + `68`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/qwen3_8_27b_q4/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `43` | `132.94 ms` | `339.57 tok/s` | **`84.37 tok/s`** | `209` (`0`T + `209`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/qwen3_8_27b_q4/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `37` | `105.93 ms` | `369.45 tok/s` | **`75.78 tok/s`** | `1242` (`41`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/qwen3_8_27b_q4/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `61` | `110.22 ms` | `581.61 tok/s` | **`63.14 tok/s`** | `672` (`284`T + `387`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/qwen3_8_27b_q4/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1868` | `1205.65 ms` | `1564.0 tok/s` | **`50.16 tok/s`** | `180` (`0`T + `180`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/qwen3_8_27b_q4/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2923` | `2436.5 ms` | `1207.35 tok/s` | **`43.15 tok/s`** | `49` (`0`T + `49`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/qwen3_8_27b_q4/P6_SCALE_3K.md) |

### Qwen 3.8 27B Vision (13G Mixed Quant + MTP) (`qwen3_8_27b_vision_13g`)
- **Architecture**: Qwen 3.8 Vision (32L ViT + 64L Hybrid Gated DeltaNet/GQA + Native MTP Self-Drafter)
- **Quantization**: 14.19 GB Mixed INT4/FP4/BF16 + Native MTP Drafter
- **Memory Footprint**: `21.97 GB` VRAM Used (`73.18 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/qwen3_8_27b_vision_13g/server_startup.log`](outputs/qwen3_8_27b_vision_13g/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `38` | `122.55 ms` | `324.37 tok/s` | **`58.99 tok/s`** | `57` (`0`T + `57`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/qwen3_8_27b_vision_13g/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `43` | `97.23 ms` | `471.99 tok/s` | **`81.95 tok/s`** | `234` (`0`T + `234`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/qwen3_8_27b_vision_13g/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `37` | `133.34 ms` | `360.64 tok/s` | **`73.56 tok/s`** | `1229` (`28`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/qwen3_8_27b_vision_13g/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `61` | `123.41 ms` | `517.77 tok/s` | **`64.06 tok/s`** | `955` (`504`T + `450`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/qwen3_8_27b_vision_13g/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1868` | `1247.9 ms` | `1509.87 tok/s` | **`49.26 tok/s`** | `180` (`0`T + `180`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/qwen3_8_27b_vision_13g/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2923` | `2225.38 ms` | `1321.64 tok/s` | **`46.67 tok/s`** | `33` (`0`T + `33`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/qwen3_8_27b_vision_13g/P6_SCALE_3K.md) |

### NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4) (`nemotron_3_5_bolt`)
- **Architecture**: Nemotron-H Puzzle (Hybrid Mamba-2 SSM + GQA + 40L LatentMoE, Top-9 of 496)
- **Quantization**: 8.27 GB BF16/FP8 Dense + 54.56 GB Native NVFP4 Experts
- **Memory Footprint**: `73.7 GB` VRAM Used (`21.46 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/nemotron_3_5_bolt/server_startup.log`](outputs/nemotron_3_5_bolt/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `27` | `79.27 ms` | `373.14 tok/s` | **`52.24 tok/s`** | `39` (`0`T + `39`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/nemotron_3_5_bolt/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `33` | `99.8 ms` | `355.57 tok/s` | **`50.46 tok/s`** | `320` (`0`T + `320`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/nemotron_3_5_bolt/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `35` | `87.8 ms` | `433.33 tok/s` | **`54.59 tok/s`** | `1280` (`79`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/nemotron_3_5_bolt/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `51` | `111.09 ms` | `485.2 tok/s` | **`51.2 tok/s`** | `488` (`245`T + `242`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/nemotron_3_5_bolt/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1866` | `1922.04 ms` | `1000.34 tok/s` | **`44.62 tok/s`** | `136` (`0`T + `136`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/nemotron_3_5_bolt/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2927` | `3215.5 ms` | `940.27 tok/s` | **`43.13 tok/s`** | `57` (`0`T + `57`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/nemotron_3_5_bolt/P6_SCALE_3K.md) |

### NVIDIA Nemotron 3.5 Bolt 24GB (74B-A9B IQ2_XXS) (`nemotron_3_5_bolt_24gb`)
- **Architecture**: Nemotron-H Puzzle (Hybrid Mamba-2 SSM + GQA + 40L LatentMoE, Top-9 of 496)
- **Quantization**: 4.98 GB Q8/NVFP4/IQ2 Dense + 24.98 GB IQ2_XXS Routed Experts
- **Memory Footprint**: `40.1 GB` VRAM Used (`55.06 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/nemotron_3_5_bolt_24gb/server_startup.log`](outputs/nemotron_3_5_bolt_24gb/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `27` | `132.4 ms` | `246.47 tok/s` | **`45.4 tok/s`** | `54` (`0`T + `54`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/nemotron_3_5_bolt_24gb/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `33` | `153.08 ms` | `269.13 tok/s` | **`46.9 tok/s`** | `320` (`0`T + `320`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/nemotron_3_5_bolt_24gb/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `35` | `132.44 ms` | `277.14 tok/s` | **`48.96 tok/s`** | `1370` (`169`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/nemotron_3_5_bolt_24gb/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `51` | `174.74 ms` | `362.48 tok/s` | **`47.76 tok/s`** | `513` (`219`T + `293`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/nemotron_3_5_bolt_24gb/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1866` | `1726.32 ms` | `1127.14 tok/s` | **`41.77 tok/s`** | `142` (`0`T + `142`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/nemotron_3_5_bolt_24gb/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2927` | `2867.04 ms` | `1052.68 tok/s` | **`38.89 tok/s`** | `53` (`0`T + `53`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/nemotron_3_5_bolt_24gb/P6_SCALE_3K.md) |

### NVIDIA Nemotron 3.5 Lightning Vision (30B-A3B NVFP4) (`nemotron_3_5_lightning`)
- **Architecture**: Nemotron-H Hybrid Mamba-2 SSM + GQA + Sparse MoE + FrankenSTIN Vision
- **Quantization**: Native NVFP4 Routed Experts + BF16 Mamba-2/GQA
- **Memory Footprint**: `26.74 GB` VRAM Used (`68.42 GB` Free) | `0.0 GB` Pinned DDR5 L2
- **Startup Log**: [`outputs/nemotron_3_5_lightning/server_startup.log`](outputs/nemotron_3_5_lightning/server_startup.log)

| Test ID | Scenario | Prompt Tok | TTFT | Prefill Speed | Decode Speed | Generated Tok | Quality Audit | Raw Output File |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **`P1_DIRECT_CHAT`** | Direct Chat (No Thinking) | `27` | `46.39 ms` | `649.76 tok/s` | **`120.18 tok/s`** | `46` (`0`T + `46`C) | **PASS (3/3 checks passed)** | [Inspect `P1_DIRECT_CHAT.md`](outputs/nemotron_3_5_lightning/P1_DIRECT_CHAT.md) |
| **`P2_CODE_FIB`** | Code Synthesis (Python) | `33` | `60.6 ms` | `595.2 tok/s` | **`112.12 tok/s`** | `320` (`0`T + `320`C) | **PASS (3/3 checks passed)** | [Inspect `P2_CODE_FIB.md`](outputs/nemotron_3_5_lightning/P2_CODE_FIB.md) |
| **`P3_HTML_CALC`** | Interactive Web UI Code | `35` | `56.02 ms` | `691.79 tok/s` | **`104.46 tok/s`** | `1403` (`202`T + `1200`C) | **PASS (3/3 checks passed)** | [Inspect `P3_HTML_CALC.md`](outputs/nemotron_3_5_lightning/P3_HTML_CALC.md) |
| **`P4_REASONING_MATH`** | Step-by-Step Reasoning | `51` | `66.55 ms` | `831.41 tok/s` | **`112.95 tok/s`** | `774` (`440`T + `333`C) | **PASS (3/3 checks passed)** | [Inspect `P4_REASONING_MATH.md`](outputs/nemotron_3_5_lightning/P4_REASONING_MATH.md) |
| **`P5_LONG_DOC`** | Long Document Analysis (~2K tok) | `1866` | `1169.74 ms` | `1671.07 tok/s` | **`97.38 tok/s`** | `150` (`0`T + `150`C) | **PASS (3/3 checks passed)** | [Inspect `P5_LONG_DOC.md`](outputs/nemotron_3_5_lightning/P5_LONG_DOC.md) |
| **`P6_SCALE_3K`** | 3K Context Prefill Scaling (~3K tok) | `2927` | `1998.05 ms` | `1543.0 tok/s` | **`92.2 tok/s`** | `28` (`0`T + `28`C) | **PASS (3/3 checks passed)** | [Inspect `P6_SCALE_3K.md`](outputs/nemotron_3_5_lightning/P6_SCALE_3K.md) |

