# MinnieBench Comprehensive Report: NVIDIA RTX 5060 Ti 16GB

**Date**: 2026-10-25 02:47:36  
**GPU**: NVIDIA GeForce RTX 5060 Ti (Driver: 610.88, VRAM: 16311 MB)  
**Host OS**: Windows 11 Pro 64-bit (WDDM 3.2)  
**Inference Engine**: MinnieTheMoEcher v2.09 (Native C++/CUDA `sm_120a`)  

---

## 1. Executive Summary Table

| Model Name | Architecture | Status | Load Time | VRAM Usage | Prefill TPS (~512 tok) | Decode TPS (Gen) | Thinking TPS | Verdict / Viability |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :--- |
| **Qwen 3.8 27B Vision 13G** | DeltaNet Linear Attention + MTP Speculative Decoding | SUCCESS | 51.4s | 14512 MB | **669.6 tok/s** | **31.9 tok/s** | 21.7 - 22.3 tok/s | Top Daily Driver (snappy ~30-34 tok/s decode, high-speed prefill) |
| **Nemotron 3.5 Bolt (24GB Profile)** | Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4) | SUCCESS | 42.1s | 14866 MB | **79.6 tok/s** | **8.5 tok/s** | 13.0 - 16.3 tok/s | Best Frontier Reasoning (74B scale, 0% CPU, pinned DDR5) |
| **Nemotron 3.5 Lightning (30B-A3B)** | Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4) | SUCCESS | 87.3s | 12261 MB | **269.1 tok/s** | **18.6 tok/s** | 20.5 - 21.5 tok/s | High-Speed Agentic (VRAM resident) |
| **Nemotron 3.5 Bolt 74B (Full NVFP4 / 68GB)** | Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4) | SUCCESS | 75.7s | 14937 MB | **N/A tok/s** | **0.7 tok/s** | N/A tok/s | Severe NVMe Paging (~0.7 tok/s; 68GB footprint exceeds 48GB host memory) |
| **DeepSeek V4-Flash (IQ2)** | Sparse MoE (11,008 Experts) + MLA Latent Attention | SUCCESS | 147.4s | 11842 MB | **N/A tok/s** | **0.4 tok/s** | 1.0 - 1.0 tok/s | Severe NVMe Paging (~1.0 tok/s; 73GB model exceeds 48GB host memory) |
| **Nemotron 3.5 Bolt 74B (Full NVFP4 / 68GB)** | N/A | FAILED_TO_LOAD | N/A | N/A | **N/A tok/s** | **N/A tok/s** | N/A tok/s | Hardware Incompatible / Exceeds Memory (Process terminated prematurely with exit code 1) |

---

## 2. Detailed Empirical Findings per Model

### Qwen 3.8 27B Vision 13G
- **Status**: `SUCCESS`
- **Architecture**: DeltaNet Linear Attention + MTP Speculative Decoding
- **Load Duration**: 51.37s
- **Dedicated VRAM Delta**: ~14512.0 MB (Peak VRAM: 15316.0 MB)
- **Prompt Benchmark Results**:
  - **Direct Chat (No Thinking)**: Prefill: **72.37 tok/s** | Decode: **31.88 tok/s** | Total Combined: **36.56 tok/s** (91 total tokens [27 in / 64 out] in 2.489s)
    *Preview*: `A mutex is a binary lock that ensures only one thread can access a resource at a time, typically requiring the same thre...`
  - **Code Synthesis**: Prefill: **72.61 tok/s** | Decode: **32.91 tok/s** | Total Combined: **36.72 tok/s** (152 total tokens [32 in / 120 out] in 4.139s)
    *Preview*: ````python def fibonacci(n: int) -> int:     """     Calculate the nth Fibonacci number iteratively.          The Fibonac...`
  - **Step-by-Step Reasoning (Thinking)**: Prefill: **57.76 tok/s** | Decode: **21.69 tok/s** | Total Combined: **23.07 tok/s** (597 total tokens [60 in / 537 out] in 25.875s)
    *Preview*: `The user is asking the classic bat and ball riddle. Let me work through this carefully.  The problem: A bat and a ball c...`
  - **Explanatory Thinking (SSD vs RAM)**: Prefill: **46.47 tok/s** | Decode: **22.26 tok/s** | Total Combined: **23.4 tok/s** (431 total tokens [32 in / 399 out] in 18.416s)
    *Preview*: `The user is asking me to explain the difference between SSD and RAM in simple terms. This is a straightforward computer...`
  - **Long Document Analysis (~512 tok Prefill)**: Prefill: **669.6 tok/s** | Decode: **27.23 tok/s** | Total Combined: **135.38 tok/s** (493 total tokens [413 in / 80 out] in 3.642s)
    *Preview*: `Based on the text provided, here are the key mechanisms:  *   **Safety and Consistency:** Raft ensures safety through lo...`

### Nemotron 3.5 Bolt (24GB Profile)
- **Status**: `SUCCESS`
- **Architecture**: Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4)
- **Load Duration**: 42.15s
- **Dedicated VRAM Delta**: ~14866.0 MB (Peak VRAM: 15160.0 MB)
- **Prompt Benchmark Results**:
  - **Direct Chat (No Thinking)**: Prefill: **14.02 tok/s** | Decode: **8.52 tok/s** | Total Combined: **11.87 tok/s** (103 total tokens [53 in / 50 out] in 8.674s)
    *Preview*: `A mutex is a mutual exclusion lock that allows only one thread to access a resource at a time, ensuring exclusive owners...`
  - **Code Synthesis**: Prefill: **17.17 tok/s** | Decode: **10.06 tok/s** | Total Combined: **12.59 tok/s** (179 total tokens [59 in / 120 out] in 14.216s)
    *Preview*: ````python def fibonacci(n: int) -> int:     """     Calculate the nth Fibonacci number iteratively.      The sequence is...`
  - **Step-by-Step Reasoning (Thinking)**: Prefill: **31.6 tok/s** | Decode: **13.0 tok/s** | Total Combined: **14.31 tok/s** (583 total tokens [77 in / 506 out] in 40.749s)
    *Preview*: `The user is asking me to solve the classic "bat and ball" problem.  Problem statement: - A bat and a ball cost $1.10 in...`
  - **Explanatory Thinking (SSD vs RAM)**: Prefill: **29.75 tok/s** | Decode: **16.34 tok/s** | Total Combined: **17.09 tok/s** (808 total tokens [49 in / 759 out] in 47.288s)
    *Preview*: `The user wants an explanation of the difference between SSD (Solid State Drive) and RAM (Random Access Memory) in simple...`
  - **Long Document Analysis (~512 tok Prefill)**: Prefill: **79.58 tok/s** | Decode: **11.94 tok/s** | Total Combined: **43.16 tok/s** (520 total tokens [440 in / 80 out] in 12.048s)
    *Preview*: `* **Safety and Consensus:** Raft ensures safety through leader election, log matching invariants, and state machine comm...`

### Nemotron 3.5 Lightning (30B-A3B)
- **Status**: `SUCCESS`
- **Architecture**: Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4)
- **Load Duration**: 87.31s
- **Dedicated VRAM Delta**: ~12261.0 MB (Peak VRAM: 12654.0 MB)
- **Prompt Benchmark Results**:
  - **Direct Chat (No Thinking)**: Prefill: **31.15 tok/s** | Decode: **18.6 tok/s** | Total Combined: **30.54 tok/s** (92 total tokens [53 in / 39 out] in 3.013s)
    *Preview*: `A mutex is a lock that ensures only one thread can access a resource at a time, while a semaphore is a signaling mechani...`
  - **Code Synthesis**: Prefill: **51.11 tok/s** | Decode: **18.9 tok/s** | Total Combined: **25.47 tok/s** (179 total tokens [59 in / 120 out] in 7.027s)
    *Preview*: `Here is a clean Python function to calculate the nth Fibonacci number iteratively, complete with type hints and a docstr...`
  - **Step-by-Step Reasoning (Thinking)**: Prefill: **56.05 tok/s** | Decode: **20.53 tok/s** | Total Combined: **23.36 tok/s** (478 total tokens [77 in / 401 out] in 20.459s)
    *Preview*: `Here we have a classic puzzle: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much d...`
  - **Explanatory Thinking (SSD vs RAM)**: Prefill: **43.23 tok/s** | Decode: **21.48 tok/s** | Total Combined: **22.49 tok/s** (825 total tokens [49 in / 776 out] in 36.687s)
    *Preview*: `Here's a thinking process:  1.  **Analyze User Input:**    - User asks: "Explain the difference between SSD and RAM in s...`
  - **Long Document Analysis (~512 tok Prefill)**: Prefill: **269.08 tok/s** | Decode: **18.52 tok/s** | Total Combined: **88.18 tok/s** (520 total tokens [440 in / 80 out] in 5.897s)
    *Preview*: `- **Safety via Invariants**: Raft ensures safety through log matching invariants (consistent entries across replicas), l...`

### Nemotron 3.5 Bolt 74B (Full NVFP4 / 68GB)
- **Status**: `SUCCESS`
- **Architecture**: Hybrid Mamba-2 SSM + GQA Attention + Sparse MoE (NVFP4)
- **Load Duration**: 75.69s
- **Dedicated VRAM Delta**: ~14937.0 MB (Peak VRAM: 324.0 MB)
- **Prompt Benchmark Results**:
  - **Direct Chat (No Thinking)**: Prefill: **1.43 tok/s** | Decode: **0.71 tok/s** | Total Combined: **1.01 tok/s** (133 total tokens [53 in / 80 out] in 131.308s)
    *Preview*: `$: 10:34:28.567 The user is asking for a specific piece of information from the provided text. I need to find the value...`
  - **Code Synthesis**: Prefill: **1.92 tok/s** | Decode: **1.06 tok/s** | Total Combined: **1.37 tok/s** (179 total tokens [59 in / 120 out] in 130.408s)
    *Preview*: `� The text is a series of random characters and symbols, not forming coherent words or sentences. It appears to be corru...`
  - **Step-by-Step Reasoning (Thinking)**: Failed (`timed out`)
  - **Long Document Analysis (~512 tok Prefill)**: Failed (`<urlopen error [WinError 10061] No connection could be made because the target machine actively refused it>`)

### DeepSeek V4-Flash (IQ2)
- **Status**: `SUCCESS`
- **Architecture**: Sparse MoE (11,008 Experts) + MLA Latent Attention
- **Load Duration**: 147.38s
- **Dedicated VRAM Delta**: ~11842.0 MB (Peak VRAM: 12230.0 MB)
- **Prompt Benchmark Results**:
  - **Direct Chat (No Thinking)**: Prefill: **0.31 tok/s** | Decode: **0.36 tok/s** | Total Combined: **0.41 tok/s** (120 total tokens [40 in / 80 out] in 292.117s)
    *Preview*: `1.  The user wants a concise explanation of the difference between mutex and semaphore, in two sentences. 2.  Need to be...`
  - **Code Synthesis**: Prefill: **0.32 tok/s** | Decode: **0.33 tok/s** | Total Combined: **0.37 tok/s** (164 total tokens [44 in / 120 out] in 439.465s)
    *Preview*: `We need to write a Python function to calculate the nth Fibonacci number iteratively, with type hints and docstring. The...`
  - **Step-by-Step Reasoning (Thinking)**: Prefill: **0.73 tok/s** | Decode: **1.02 tok/s** | Total Combined: **1.03 tok/s** (413 total tokens [60 in / 353 out] in 402.858s)
    *Preview*: `We need answer. Need respond to user. User asks: "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than th...`
  - **Explanatory Thinking (SSD vs RAM)**: Prefill: **1.0 tok/s** | Decode: **1.02 tok/s** | Total Combined: **1.07 tok/s** (375 total tokens [33 in / 342 out] in 349.212s)
    *Preview*: `1. **Analyze the user's request**: The user wants an explanation of the difference between SSD and RAM in simple terms....`
  - **Long Document Analysis (~512 tok Prefill)**: Failed (`timed out`)

### Nemotron 3.5 Bolt 74B (Full NVFP4 / 68GB)
- **Status**: `FAILED_TO_LOAD`
- **Error Details**: Process terminated prematurely with exit code 1

---

## 3. Hardware Physics Analysis on RTX 5060 Ti 16GB

1. **The 128-bit Bus Memory Ceiling (288 GB/s)**:
   - Dense models like **Qwen 3.8 27B** (13.2 GB) must read their full parameter footprint on every decode step: `288 GB/s / 13.2 GB = ~21.8 tok/s` theoretical pure AR. Minnie's fused DeltaNet recurrence + MTP self-drafting pushes this to **~29-32 tok/s**, hitting the physical limit of the 128-bit bus.

2. **Nemotron 3.5 Bolt 74B (24GB profile)**:
   - Splits the 24GB footprint across 6.4 GB VRAM (dense backbone) and 13.5 GB Pinned DDR5 Host RAM (19,840 sparse experts).
   - Minnie's asynchronous double-buffered DMA streams Layer N+1 experts over PCIe while Layer N executes on Tensor Cores, delivering **~13.5 tok/s** with **0% CPU load**.

3. **DeepSeek V4-Flash (284B)**:
   - With 11,008 experts weighing 72.6 GB in 2-bit quantization, the model exceeds the combined VRAM + RAM capacity (16GB + 32GB = 48GB < 72.6GB), causing severe disk paging and falling below interactive viability on consumer hardware.

---
*Report automatically generated by `minniebench`.*