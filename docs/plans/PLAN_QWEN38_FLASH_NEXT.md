# Qwen3.8-Flash-Next (`Qwen4Exp`) Implementation & Optimization Plan

## 1. Overview & Model Target
**`Qwen3.8-Flash-Next`** (`architectures: ["Qwen4ExpForConditionalGeneration"]`, `model_type: "qwen4_exp"`) is Qwen's experimental Qwen 4.0 architecture preview:
- **Total Parameters**: 180B (`125B` Language Model with **`6B` active per token** + `51B` Layer-2 N-Gram PLE table + `4B` 1-layer MTP)
- **Layers**: `48` (`12 × [3 × Gated DeltaNet + 1 × Qwen Sparse Attention (QSA)]`)
- **Hidden Size**: `2560` | **Vocab Size**: `248,320`
- **Gated Residual (4-Branch Hyper-Connections)**: `hc_count = 4`, bottleneck rank `hc_lowrank = 320` (element-wise data-dependent read gate + per-branch scalar write gate)
- **Gated DeltaNet (36 layers)**: `16` QK heads (`head_dim = 128`), `48` V heads (`head_dim = 128`), `conv_kernel = 4`
- **Qwen Sparse Attention / QSA (12 layers — layers 3, 7, 11, ..., 47)**:
  - `24` Q heads, `2` KV heads, `head_dim = 256`, `partial_rotary_factor = 0.25` (`rope_dim = 64`, `mrope_section = [11, 11, 10]`, `rope_theta = 10,000,000`)
  - **Micro-Block MQA Indexer**: `4` Q heads, `1` KV head, `indexer_head_dim = 128`, `indexer_compress_ratio = 4`, `indexer_budget = 2048` tokens (`512` blocks)
- **Mixture-of-Experts (All 48 layers)**:
  - Unpruned: `512` routed experts/layer (`GSQ-RCO-Coder` retains `256` RCO-selected experts/layer)
  - Active per token: `10` routed experts + `1` shared expert (`moe_intermediate_size = 640`, `shared_expert_intermediate_size = 640`)
- **Layer-2 N-Gram PLE Lookup Table (`ple_layer_ids = [2]`)**:
  - `ngram_vocab_size_base = 20,000,000` (bigrams & trigrams, `ngram_size = 3`), `heads_per_ngram = 8`, `ple_embed_dim = 2560`, `ple_conv_kernel_size = 4`
  - Zero matrix-multiply storage scaling: deterministic hash lookup from input token IDs, reading `< 16 KB` per token at Layer 2.

---

## 2. Primary Quantized Checkpoint (`ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF`)
Selected to run **100% in memory (zero SSD expert reads)** on **RTX 5060 Ti 16GB + 32GB System RAM (12GB Windows WDDM Pinned Limit)** while also running **100% in VRAM** on a single **RTX PRO 6000 Blackwell 96GB** (and sharing the exact same engine/layout as the unpruned 512-expert `GSQ-RCO IQ3_XXS` / `IQ3_S` packs):

| Component | File | Size | Placement on **RTX 5060 Ti 16GB (32GB RAM)** | Placement on **RTX PRO 6000 96GB** |
| :--- | :--- | :---: | :--- | :--- |
| **Shard 1 (Transformer + 256/512 Experts @ 3.5 bpw)** | `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf` | **27.58 GiB** (`29.6 GB`) | **12.5 GiB VRAM L1** + **11.5 GiB Pinned RAM L2a** + **3.58 GiB Unpinned RAM L2b** | **100% in VRAM L1** (`27.58 GiB`) |
| **Shard 2 (20M N-Gram PLE Table @ `IQ4_NL` 4.5 bpw)** | `Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf` | **26.82 GiB** (`28.8 GB`) | **NVMe SSD L3** (`DirectFileHandle` + `EngramRowCache` prefetch at Layer 0) | **100% in VRAM L1** or **Pinned RAM** (`26.82 GiB`) |
| **Vision Projector (`mmproj`)** | `mmproj-Qwen3.8-Flash-Next-BF16.gguf` | **0.85 GiB** (`0.91 GB`) | Loaded on demand for multimodal input | Resident in VRAM |
| **Benchmarks vs 354 GB BF16** | — | **55.25 GiB** | **100.0% LiveCodeBench v6 (`87.43`)**, **95.7% SWE-bench Verified (`79.20`)** | Same (or swap Shard 1 for `IQ3_S` `54.8 GB` for 100.1% general recovery) |

---

## 3. Strata Analysis: What We Keep vs. What We Port

### A. What We Explicitly Reject from Strata
- **CPU Expert Compute (`pcie_share_rule.hpp` / `iq_avx2.hpp`) — REJECTED:**
  1. **DDR5 Memory Bus Contention:** Consumer dual-channel DDR5 (~55–65 GB/s real) is already saturated by PCIe 4.0/5.0 H2D DMA (`25–50 GB/s`). Waking CPU threads to dequantize LUT-based `IQ2`/`IQ3` experts steals memory controller bandwidth from PCIe DMA.
  2. **AVX-512 Hardware Fragility & Straggler Stalls:** Without AVX-512 VBMI, low-bit LUT dequantization on CPU cores stalls the GPU barrier at the end of every layer.
  3. **WDDM & Subsystem Starvation:** Keeping **0% CPU compute burn** guarantees sub-microsecond CUDA stream dispatch under Windows WDDM and leaves CPU cores free for Moecher's C++ Agentic Loop, MCP servers, voice, and Web UI.

### B. What MinnieTheMoEcher Already Has
- **32-Slot Pinned Staging Ring Buffer (`staging_ring_[32]`) & 8-Stream DMA Staging (`dma_staging_[8]`)** in [`src/expert_loader.hpp`](file:///e:/dev/MinnieTheMoEcher/src/expert_loader.hpp#L83-L98).
- **Adaptive 512 MB Step-Down Pinned RAM Probing** in [`src/expert_loader.hpp`](file:///e:/dev/MinnieTheMoEcher/src/expert_loader.hpp#L563-L570).
- **Layer-0 Async SSD N-Gram Prefetch + 65,536-Slot `EngramRowCache`** in [`src/engine.hpp`](file:///e:/dev/MinnieTheMoEcher/src/engine.hpp#L770-L940).
- **4-Stream Hyper-Connections (`buf_hc_state_`) & Sparse Top-512 Indexer** in [`src/models/model_deepseek_v41.inl`](file:///e:/dev/MinnieTheMoEcher/src/models/model_deepseek_v41.inl).

### C. What We Port / Adapt into MinnieTheMoEcher
1. **Dual-Pool L2 RAM (`Pinned L2a` + `Unpinned Overflow L2b` via `dma_staging_`) in [`src/expert_loader.hpp`](file:///e:/dev/MinnieTheMoEcher/src/expert_loader.hpp):**
   - When `cudaMallocHost` reaches Windows WDDM's ~12 GB pinned ceiling (`L2a`), allocate the remaining requested L2 budget (~3.6–5 GB) via unpinned `VirtualAlloc` (`L2b`).
   - On an `L2b` hit, perform a ~7 µs DDR5 `memcpy` into our existing `dma_staging_[k]` pinned buffer and immediately fire `cudaMemcpyAsync` to VRAM.
   - **Impact:** Keeps 100% of `GSQ-RCO-Coder` Shard 1 (`27.58 GiB`) in `VRAM + RAM` on a 16GB VRAM / 32GB RAM Windows PC with zero SSD reads for experts.
2. **Idle-PCIe Attention-Phase Expert Prefetch (`RouterLookahead`):**
   - Because `Qwen3.8-Flash-Next` has a very small Shared Expert (`640` intermediate dim), the Shared Expert alone is too fast to hide PCIe DMA of 10 missed routed experts.
   - By evaluating a lightweight router lookahead at the start of the layer's DeltaNet / QSA Attention phase (when the PCIe bus is 100% idle), we pre-DMA L2-resident experts into VRAM while Attention executes on the GPU.
3. **Opt-In `RoutePrior` (`route_prior_top10` Cache-Conditioned Routing):**
   - Adds an optional small logit bonus `+λ` (`default = 0.0` for 100% exact math; configurable `0.05–0.10` for Turbo Offload Mode on 16GB GPUs) to VRAM-resident experts during Top-10 selection, while renormalizing the *unbiased* router softmax probabilities over the selected 10. Automatically disabled when 100% VRAM resident (e.g., on RTX PRO 6000 96GB).
4. **Phase 2 Speculative Upgrade — `DFlash2` 7-Token Dense Block Drafter (`incoai/Qwen3.8-Flash-Next-DFlash2`):**
   - Unlike the built-in 1-layer MTP head (which has 512 routed MoE experts = 4B params), `DFlash2` is a 5-layer **dense** transformer (~1.1 GB in Q8, zero MoE experts) that taps hidden states from layers `5, 15, 25, 35, 45` and drafts 7 tokens in a single parallel forward pass inside VRAM.

---

## 4. Implementation Milestones
- [x] **Step 1:** Create branch `models/qwen3.8-flash-next` and store this plan in `PLAN_QWEN38_FLASH_NEXT.md`.
- [ ] **Step 2:** Download `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` (Shard 1 `27.58 GiB`, Shard 2 `26.82 GiB`, `mmproj` `0.85 GiB`, and RCO allocation manifest) + `Qwen/Qwen3.8-Flash-Next` tokenizer/config files into `models/qwen3.8_flash_next/`.
- [ ] **Step 3:** Inspect exact GGUF tensor names, shapes, and GGML quant types (`Q6_K`, `Q5_K`, `Q4_K`, `IQ4_NL`, `IQ4_XS`, `IQ3_S`, `IQ3_XXS`, `IQ2_S`, `Q2_0`, `BF16`, `F32`) in Shard 1 and Shard 2.
- [ ] **Step 4:** Upgrade [`src/expert_loader.hpp`](file:///e:/dev/MinnieTheMoEcher/src/expert_loader.hpp) with Dual-Pool L2 RAM (`Pinned L2a` + `Unpinned VirtualAlloc L2b` bouncing through `dma_staging_`).
- [ ] **Step 5:** Implement `Qwen4Exp` CUDA kernels in [`src/cuda/activations.cu`](file:///e:/dev/MinnieTheMoEcher/src/cuda/activations.cu) / [`src/cuda/activations.cuh`](file:///e:/dev/MinnieTheMoEcher/src/cuda/activations.cuh):
  - 4-Branch Low-Rank Gated Residual (`hc_count = 4`, `hc_lowrank = 320`: `hc_attn_down`, `hc_attn_up`, `hc_attn_inject`, `hc_attn_norm`).
  - Gated DeltaNet (`16` QK heads, `48` V heads, `head_dim = 128`, `conv_kernel = 4`).
  - Qwen Sparse Attention (`QSA`: `24` Q heads, `2` KV heads, `head_dim = 256`, `rope_dim = 64` + micro-block MQA Indexer with `4` Q / `1` KV head, `budget = 512` blocks).
  - Layer-2 PLE N-Gram lookup (`20M × 2560` `IQ4_NL` Shard 2 reader + `ple_conv1d`, `ple_key`, `ple_value`, norms).
  - Multi-quant MoE expert execution (`IQ3_XXS`, `IQ3_S`, `IQ2_S`, `IQ4_NL`, `Q2_0`) + Top-10 Sigmoid/Softmax Router + `RoutePrior`.
- [ ] **Step 6:** Implement [`src/models/model_qwen4.inl`](file:///e:/dev/MinnieTheMoEcher/src/models/model_qwen4.inl) and wire into [`src/engine.hpp`](file:///e:/dev/MinnieTheMoEcher/src/engine.hpp), [`src/main.cpp`](file:///e:/dev/MinnieTheMoEcher/src/main.cpp), and [`src/server.cpp`](file:///e:/dev/MinnieTheMoEcher/src/server.cpp).
- [ ] **Step 7:** End-to-end build, numerical verification, and performance benchmarking on RTX 5060 Ti 16GB.

