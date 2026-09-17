# MinnieTheMoECher: Engineering, Research & Discoveries Log

A comprehensive chronological record of engineering breakthroughs, mathematical analyses, performance bottlenecks, root-cause investigations, and architectural milestones in **MinnieTheMoECher**.

---

## Table of Contents
1. [Endeavour 1: DeepSeek V4-Flash 100% Resident MoE Quantization (IQ2_XXS + Q2_K)](#endeavour-1-deepseek-v4-flash-100-resident-moe-quantization-iq2_xxs--q2_k)
2. [Endeavour 2: Context Retention & Compressed Sequence Attention (CSA & HCA)](#endeavour-2-context-retention--compressed-sequence-attention-csa--hca)
3. [Endeavour 3: Qwen 3.8 27B Native MTP Speculative Decoding (~98 tok/s Milestone)](#endeavour-3-qwen-38-27b-native-mtp-speculative-decoding-98-toks-milestone)
4. [Endeavour 4: Mixed INT3/INT4 Quantization for 16GB GPUs (`qwen3.8-27B-Vision-13G`)](#endeavour-4-mixed-int3int4-quantization-for-16gb-gpus-qwen38-27b-vision-13g)
5. [Endeavour 5: Recurrent State Preservation — The DeltaNet SSM Rollback Overflow](#endeavour-5-recurrent-state-preservation--the-deltanet-ssm-rollback-overflow)
6. [Endeavour 6: Speculative Verification Dequantization Thrashing (14 tok/s $\to$ 110 tok/s)](#endeavour-6-speculative-verification-dequantization-thrashing-14-toks--110-toks)
7. [Endeavour 7: Server Streaming Latency, Nagle's Algorithm (`TCP_NODELAY`), and Zero-Alloc SSE](#endeavour-7-server-streaming-latency-nagles-algorithm-tcp_nodelay-and-zero-alloc-sse)
8. [Endeavour 8: Context-Length Scaling Bottleneck — 3,145-Token Tool Attention (44 tok/s vs 109 tok/s)](#endeavour-8-context-length-scaling-bottleneck--3145-token-tool-attention-44-toks-vs-109-toks)
9. [Endeavour 9: Pinned System KV Snapshots & Autonomous Agentic Suite](#endeavour-9-pinned-system-kv-snapshots--autonomous-agentic-suite)
10. [Roadmap of Pending Optimizations](#roadmap-of-pending-optimizations)

---

## Endeavour 1: DeepSeek V4-Flash 100% Resident MoE Quantization (IQ2_XXS + Q2_K)

### Problem Statement
DeepSeek V4-Flash contains **11,008 MoE experts** across 43 layers. The uncompressed FP4 weights required **157.4 GB**, exceeding 96GB VRAM on RTX PRO 6000 (Blackwell). NVMe SSD streaming and DRAM offloading resulted in disk I/O stalls, capping decode throughput at **~35 tok/s**.

### Discovery & Solution
1. **Calibrated Multi-Format Quantization**:
   - Quantized expert gate/up matrices ($W_1, W_3$) to **`IQ2_XXS`** (2.0625 bpw) using importance matrix (`imatrix`) calibration.
   - Quantized expert down matrices ($W_2$) to **`Q2_K`** (2.5625 bpw) with $16 \times 16$ nested scales.
   - Result: Expert footprint plummeted from **157.4 GB $\to$ 72.56 GB**, enabling **100% VRAM residency** of all 11,008 experts with 18+ GB left for KV cache and dense layers.
2. **GPU Top-6 Routing Reduction**:
   - Replaced host-side `std::partial_sort` (which caused blocking D2H synchronization stalls) with an in-place single-block GPU reduction kernel.
3. **Asynchronous Multi-Stream Overlap**:
   - Overlapped Q and KV attention projections on separate CUDA streams, while overlapping routed MoE evaluation with shared expert GEMVs.
4. **Fused SwiGLU MoE Kernel**:
   - Fused $W_1$ (Gate) and $W_3$ (Up) GEMV + `SiLU(gate) * up` into a single CUDA dispatch per active expert.

### Performance Impact
- Decode speed jumped from **35.2 tok/s $\to$ 53.98 tok/s** (+53.4%).
- Time-to-First-Token (TTFT) improved from **1.4s $\to$ 694ms** (~2x faster).

---

## Endeavour 2: Context Retention & Compressed Sequence Attention (CSA & HCA)

### Problem Statement
During multi-step reasoning deliberation (`<think>...</think>`), the model frequently suffered reasoning context amnesia after ~32 tokens, looping or forgetting premises.

### Root Cause Analysis
1. **Compressor Cache Capacity Cap**:
   - In the compressed KV cache implementation, storage capacity was inadvertently hardcoded to `min(32, max_seq_len / ratio)`, capping history retention to 32 compressed tokens regardless of available VRAM.
2. **Attention Softmax Scale Anomaly**:
   - A redundant scaling factor was applied during QK dot products in addition to $1/\sqrt{d_{\text{head}}}$, diluting attention sharpness across long contexts.
3. **RoPE YaRN Over-Scaling**:
   - RoPE coordinate calculation applied a $(1.277)^2$ magnitude boost instead of unit scaling ($mscale = 1.0$).

### Fix & Impact
- Expanded the CSA/HCA dynamic circular buffers across the full 32,768-token sequence context.
- Fixed Softmax and RoPE scaling constants to match reference transformer formulations.
- Result: Completely eliminated reasoning context amnesia and enabled coherent 4,000+ token thinking traces.

---

## Endeavour 3: Qwen 3.8 27B Native MTP Speculative Decoding (~98 tok/s Milestone)

### Problem Statement
Autoregressive decoding of Qwen 3.8 27B (hybrid DeltaNet linear attention + full GQA) ran at **55.0 tok/s**. Attempting to use an external 2B neural drafter actually **degraded** speed to **42.0 tok/s** (-24%) due to high draft latency (8 ms) and poor token acceptance (22.1%).

### Breakthrough Discovery
1. **Native MTP Self-Drafting**:
   - Qwen 3.8 includes an integrated **Multi-Token Prediction (MTP)** layer (`mtp.layers.0`, `mtp.fc`). By leveraging the target model's own representations, draft tokens share the exact hidden context of the model.
2. **Compact 40,000-Token Draft Vocabulary (`draft_vocab_ids.bin`)**:
   - Rather than projecting across all 248,077 vocabulary tokens, we identified that the top 40,000 BPE tokens account for >99.2% of real-world text and code.
   - Built a 390.6 MB BF16 projection head (`draft_lm_head_int8_bf16.bin`).
   - Draft latency dropped from **8.0 ms $\to$ 1.18 ms** per cycle (6.5x reduction).
3. **Batched Shared-Memory DeltaNet Recurrence (`deltanet_ssm_batch_kernel`)**:
   - In Qwen's 48 linear attention layers, verifying $M$ candidate tokens previously required copying hidden states back and forth between global VRAM buffers.
   - We implemented a fused kernel maintaining the SSM recurrence state across all $M$ steps entirely in **32 KB on-chip shared memory**.
   - Eliminated **360 D2D memcpys** and **188 MB** of intermediate VRAM roundtrips per cycle.
4. **Post-Norm Hidden State Propagation**:
   - Discovered that passing un-normalized hidden states into the MTP projection caused MTP acceptance to collapse to 2.6%. Propagating post-norm states restored acceptance to **62.0% – 73.6%**.

### Performance Impact
- Verified sustained speed reached **97.94 tok/s** (+78% over AR baseline, +133% over 2B drafter).

---

## Endeavour 4: Mixed INT3/INT4 Quantization for 16GB GPUs (`qwen3.8-27B-Vision-13G`)

### Problem Statement
Standard INT4 checkpoints of Qwen 3.8 27B occupied **>20 GB**, exceeding the VRAM of mainstream 16GB cards (RTX 4060 Ti, RTX 5060 Ti) and forcing PCIe host memory paging that collapsed generation to single digits.

### Architecture Design: `qwen3.8-27B-Vision-13G`
We designed a selective precision hierarchy implemented in [`scripts/quantize_qwen_vision.py`](file:///home/tinobruno/minniethemoecher/scripts/quantize_qwen_vision.py):

| Component | Target Mode | Size | Purpose |
| :--- | :--- | :--- | :--- |
| **Visual Tower** (`model.visual.*`) | **BF16 Raw** | 0.92 GB | 100% OCR fidelity and vision perception. |
| **Embeddings & LM Head** | **INT4 Block-32** | 1.44 GB | Saves 3.64 GB over BF16 while preserving vocabulary entropy. |
| **Linear & Self Attention** | **INT4 Block-32** | 3.17 GB | Guards DeltaNet recurrence numerical stability. |
| **MLP Projections** (`gate/up/down`) | **INT3 Block-32** (3.50 bpw) | 7.49 GB | Reduces 17.1B parameters from 10.16 GB $\to$ 7.49 GB. |
| **Norms & Biases** | **BF16 / FP32** | 0.10 GB | Zero precision drift in layer norms. |
| **Total** | | **~13.1 GB** | **Leaves ~2.8 GB headroom on 16 GB GPUs.** |

---

## Endeavour 5: Recurrent State Preservation — The DeltaNet SSM Rollback Overflow

### Problem Statement
During complex reasoning prompts (e.g. *"Which number is bigger: 9.11 or 9.9?"*), the model's thinking stream began hallucinating numbers and devolved into an infinite repetitive loop (`24.1.2.1.2.1.1.1.1.1.1...`).

### Root Cause Analysis
1. In Qwen 3.8, the 48 Linear Attention layers maintain recurrent state tuples $(\text{SSM state}, \text{Conv state})$.
2. During speculative verification, the CUDA kernel preserved backup states in rollback checkpoint slots. However, only **2 rollback slots** (`slot 0` and `slot 1`) were allocated in `target_ssm_pool_`.
3. Prompt-Lookup Drafting (PLD) was proposing 4 to 5 candidate tokens ($M = 5$ or $6$).
4. When verification accepted $\ge 2$ tokens, `commit_target_state_slot(accepted)` attempted to restore from `slot_idx >= 2`.
5. Slots 2+ pointed to uninitialized zero memory, **wiping the recurrent state across all 48 DeltaNet layers mid-generation**.
6. With attention memory wiped, the model suffered instantaneous amnesia, looping on identical token logits.

### Fix
- Implemented a strict boundary guard in [`src/server_single.cpp`](file:///home/tinobruno/minniethemoecher/src/server_single.cpp): `if (slot_idx >= 2) return;`.
- Clamped PLD candidate length to 2 tokens for Qwen architecture: `std::min(pld_draft_tokens_, 2)`.
- Fixed MTP self-drafter weight loading for INT4 `mtp.fc.weight`.
- Result: Fixed reasoning degradation completely; 100% correct step-by-step logic on mathematical and coding benchmarks.

---

## Endeavour 6: Speculative Verification Dequantization Thrashing (14 tok/s $\to$ 110 tok/s)

### Problem Statement
After deploying the INT3/INT4 model, throughput on short prompts ("hello") collapsed down to **~14 tok/s**.

### Profiling Breakdown
- For single-token decode ($M=1$), INT3 executed via fast `gemv_int3_cuda`.
- For batched speculative verification ($M \in [2, 8]$ tokens drafted by MTP), the INT3 MLP layers had no batched kernel and fell back to `gemm_int3_dequant()`.
- `gemm_int3_dequant()` dequantized the entire 178 MB matrix into a temporary BF16 VRAM buffer on every call before launching cuBLAS.
- With 64 layers $\times$ 3 MLP projections, each verification cycle performed **192 full matrix dequantizations**, writing and reading **34.2 GB of VRAM per cycle**!
- Verification cycle latency spiked to **58.23 ms / cycle**.

### Solution: Fused & Batched INT3 Register-Unpacking Kernels
Implemented direct packed INT3 kernels in [`src/cuda/activations.cu`](file:///home/tinobruno/minniethemoecher/src/cuda/activations.cu):
1. **`gemv_int3_swiglu_fused_cuda`** ($M=1$): Fuses Gate and Up INT3 projections and `SiLU(gate) * up` directly in registers.
2. **`gemv_int3_residual_cuda`** ($M=1$): Computes Down projection and adds residual in-place.
3. **`gemm_int3_swiglu_fused_batch_cuda`** ($M \le 8$): Batched fused SwiGLU loading packed INT3 weights once and computing dot products across all $M$ tokens in registers.
4. **`gemm_int3_batch_cuda`** ($M \le 8$): Batched Down projection directly from packed INT3 weights.

### Performance Impact
- Verify cycle latency dropped from **58.23 ms $\to$ 19.97 ms** (2.92x faster).
- VRAM traffic dropped from **34.2 GB $\to$ 6.3 GB** (5.4x reduction).
- Short prompt generation jumped from **14.0 tok/s $\to$ 109.41 tok/s** (7.8x speedup).

---

## Endeavour 7: Server Streaming Latency, Nagle's Algorithm (`TCP_NODELAY`), and Zero-Alloc SSE

### Problem Statement
While CLI benchmarks achieved 109 tok/s, the Web UI and HTTP client registered jitter and lower perceived token delivery rates.

### Discovery & Fix
1. **Linux Nagle's Algorithm & Delayed ACKs**:
   - Each token chunk sent via Server-Sent Events (SSE) is tiny (~30–80 bytes).
   - The server socket had `TCP_NODELAY` disabled by default, causing the Linux network stack to buffer packets and wait up to **40 ms** for TCP ACK confirmation before transmitting the next token.
   - Fix: Added `svr.set_tcp_nodelay(true)` in [`src/server_single.cpp`](file:///home/tinobruno/minniethemoecher/src/server_single.cpp).
2. **Zero-Allocation SSE Formatting**:
   - Replaced dynamic `nlohmann::json` AST creation and serialization per token with a pre-allocated stack/buffer string formatter (`send_sse_delta`).
3. **Persisted Telemetry in Quiet Mode**:
   - Modified `log_msg()` so `--quiet` suppresses noisy per-step terminal spam while reliably persisting `[GENERATION STATS]` and `[SPECULATIVE STATS]` into `moecher.log`.

---

## Endeavour 8: Context-Length Scaling Bottleneck — 3,145-Token Tool Attention (44 tok/s vs 109 tok/s)

### Problem Statement
Testing `"hello"` via CLI produced **109.41 tok/s**, but testing `"hello"` in the Web UI produced **43.98 tok/s**.

### Root Cause Analysis
1. **Tool Schema System Prompt Injection**:
   - In CLI `--test-prompt "hello"`, prompt length is **21 tokens**.
   - In the Web UI, **Agentic & Tools** is enabled by default. The client sends `tools: "default"`, injecting full JSON schemas for 13 tools into the system prompt.
   - Total prompt length is **3,156 tokens**.
2. **GQA Attention Scaling across 16 Full-Attention Layers**:
   - Qwen 3.8 has 64 layers: 48 are linear DeltaNet layers (constant $O(1)$ step latency), but **16 are full Grouped Query Attention (GQA)** layers.
   - For every verification cycle ($M=3$ candidates), each of the 16 GQA layers computes attention across the entire 3,156-token KV cache ($3 \times 16 = 48$ kernel calls per cycle).
   - In [`qwen_gqa_compute_attn_fp8_kernel`](file:///home/tinobruno/minniethemoecher/src/cuda/activations.cu#L6216-L6228), Loop E performs sequential scalar byte reads from global memory:
     ```cuda
     for (int j = 0; j < chunk_len; j++) {
         acc0 += weight * s_fp8_lut[v_ptr[dim0]];
     }
     ```
   - Each thread performs 128 sequential global memory load transactions per chunk $\times$ 25 chunks = 3,200 memory roundtrips per kernel.
   - This adds **22.2 ms of memory-stalled latency** to every verify cycle (jumping from 19.9 ms to 42.1 ms).
3. **Early EOS Teardown Penalty**:
   - On a simple greeting ("hello"), the model finishes after only ~34 tokens. Speculative drafting only executes ~18 cycles, so startup time and candidate rejection at `<|im_end|>` depress the average speed.

### Performance Breakdown

| Metric | CLI `--test-prompt "hello"` | Web UI `"hello"` with Tools |
| :--- | :--- | :--- |
| **Prompt Length** | **21 tokens** | **3,156 tokens** (+13 tool schemas) |
| **Verify Latency** | **19.9 ms / cycle** | **42.1 ms / cycle** (+22.2 ms in GQA) |
| **Sustained Speed** | **109.41 tok/s** | **43.98 tok/s** |

### Solution: Shared-Memory Cooperative V Tiling & Batched Multi-Candidate GQA

We designed and implemented two fused optimizations in [`src/cuda/activations.cu`](src/cuda/activations.cu) and [`src/server_single.cpp`](src/server_single.cpp):

1. **Shared-Memory Cooperative V Tiling (`qwen_gqa_compute_attn_fp8_batch_kernel`)**:
   - Replaced the 128 sequential global memory byte loads per chunk with **cooperative 128-bit (`uint4`) vector loading** into `__shared__ alignas(16) uint8_t s_v_tile[32768]`.
   - All 128 threads in the block cooperatively load the $128 \times \text{head\_dim}$ chunk using coalesced 128-bit memory instructions (only **8 to 16 vector loads per thread**).
   - The inner accumulation loop unrolls directly from **L1 shared memory** with zero global memory latency stalls and zero bank conflicts:
     ```cuda
     #pragma unroll 8
     for (int j = 0; j < chunk_len; j++) {
         float weight = s_tile_scores[j];
         int j_base = j << j_shift;
         if (dim0 < head_dim) acc0 += weight * s_fp8_lut[s_v_tile[j_base + dim0]];
         if (dim1 < head_dim) acc1 += weight * s_fp8_lut[s_v_tile[j_base + dim1]];
     }
     ```
2. **Batched Multi-Candidate GQA Execution (`gridDim = (heads, M)`)**:
   - Implemented `qwen_gqa_write_kv_fp8_batch_kernel` and `qwen_gqa_compute_attn_fp8_batch_kernel`, merging candidate verification into a 2D grid `dim3(n_heads, M)`.
   - Replaced the sequential host loop in `forward_layer_qwen_batch` with a single dispatch to `qwen_gqa_decode_gated_fp8_batch_cuda`.
   - Slices kernel dispatches by 66% (from 96 launches down to 32 launches per cycle) and increases SM utilization from 11% to 34%+ across the 142 SMs on RTX PRO 6000 Blackwell.

### Verified Benchmark Results

| Metric | Before Optimization | After Optimization | Improvement |
| :--- | :--- | :--- | :--- |
| **Verify Latency (3k context)** | **42.14 ms / cycle** | **27.39 ms / cycle** | **-14.75 ms / cycle (-35.0%)** |
| **GQA Attention Overhead** | 22.2 ms / cycle | **8.4 ms / cycle** | **2.64x faster attention** |
| **Web UI "hello" (3,156 tokens)** | 43.98 tok/s | **71.52 tok/s** | **+62.6% faster** |
| **Reasoning Prompt (3,178 tokens)** | 39.05 tok/s | **67.14 tok/s** | **+71.9% faster** |
| **CLI Test Prompt (43 tokens)** | 109.41 tok/s | **111.21 tok/s** | **Peak efficiency** |
| **Startup Prefill (3,145 tokens)** | 119.70 tok/s (26.28s) | **180.97 tok/s (17.38s)** | **+51.2% faster prefill** |

---

## Endeavour 9: Pinned System KV Snapshots & Autonomous Agentic Suite

### Features & Capabilities
1. **Startup KV Cache Snapshotting**:
   - At server startup, the 3,145-token tool prompt is pre-evaluated once and pinned in memory (`Pinned System KV Cache snapshot`).
   - Subsequent user requests reuse the cached prefix, achieving **0 ms prefill latency** on turn 1.
2. **Autonomous Tool Execution (`src/tool_exec.hpp`)**:
   - Implemented native C++ tools for `read_file`, `write_file`, `edit_file`, `execute_command`, and `fetch_url`.
   - Workspace boundary enforcement (`--workspace-boundary-enforced`) preventing unauthorized traversal outside project root.
3. **Model Context Protocol (MCP) Integration**:
   - Built an MCP client (`src/mcp/`) supporting Stdio and SSE transports for external tool server integration.
4. **Dual Forward/Reverse Proxy (`port 8002`)**:
   - Transparent CORS unblocking and PAC system proxy scripting for seamless web scraping and session impersonation.

---

## Roadmap of Pending Optimizations

1. **Tensor Core / MMA Attention for GQA Decode**:
   - Explore FP8 tensor core HGEMM / MMA instructions for QK dot products and score-value accumulation on long sequence tiles ($>8\text{k}$ tokens).
2. **Expanded SSM Rollback Checkpoint Slots ($M \le 8$)**:
   - Expand `target_ssm_pool_` rollback buffers in shared memory to allow higher speculative drafting depths ($K \ge 4$) on Qwen 3.8.
3. **Zero-Copy Speculative KV Rollback**:
   - Maintain a hardware-tracked position pointer rather than overwriting rejected positions in VRAM.
