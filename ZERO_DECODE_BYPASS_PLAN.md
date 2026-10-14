# Zero Decode Bypass (ZDB) — Engineering Plan & Architecture Specification

**Branch:** `feature/ZeroDecodeBypass`  
**Reference:** *"Demystifying 'System 1' Foundation Models: Reproducing Sub-50ms Non-Autoregressive Speeds in MinnieTheMoEcher"* (Tino Bruno, Sept 2026)

---

## 1. Executive Summary & Objective

In autoregressive inference, generating structured decisions, routing tokens, and JSON classifications incurs 30 to 80 sequential forward passes through the decode loop. Each token requires streaming the full parameter weight matrix (13+ GB on 27B models) through the GPU memory bus, resulting in 400ms to 2,500ms end-to-end latency.

**Zero Decode Bypass (ZDB)** eliminates the memory-bandwidth decode bottleneck completely for decision, routing, and classification workloads:
1. **Prefill-Boundary Step-$N$ Logit Slicing**: Evaluates candidate choice logits directly at step $N$ from `buf_logits_` with **0 subsequent GPU memory sweeps**.
2. **Grammar-Guided Direct CPU Splice (GGJD)**: Stitches static schema tokens and the winning choice directly in CPU memory without GPU kernel launches.
3. **Sub-50ms Non-Autoregressive Speeds**: Reduces end-to-end latency to the raw prompt prefill duration (~15ms – 40ms on RTX 5060 Ti / Blackwell), achieving an empirical **10x – 50x speedup** on standard open weights (Qwen 3.8 / DeepSeek).

---

## 2. Architecture & Mathematical Formulation

### 2.1 Mechanical Comparison

```
[ Traditional Autoregressive Decode (ZDB Disabled) ]
Prompt Prefill (1..N) ──> Decode 1 ──> Decode 2 ──> ... ──> Decode K ──> Output Buffer
       GEMM (Compute)      GEMV (BW)    GEMV (BW)           GEMV (BW)
   [15-30ms prefill]      [~15ms/tok * 40 tokens = 600ms]                Total: ~630ms

[ Zero Decode Bypass (ZDB Enabled) ]
Prompt Prefill (1..N) ──> Step N Logit Slicing ──> CPU Direct Splice ──> Output Buffer
       GEMM (Compute)        cudaMemcpyAsync (Host)       memcpy (Host)
   [15-30ms prefill]              [<0.05ms]                  [<0.01ms]     Total: ~20-30ms
```

### 2.2 Sliced Energy & Probability Extraction
Given a prompt terminating at token index $N$ and a discrete candidate choice set:
$$\mathcal{C} = \{c_0, c_1, \dots, c_{K-1}\}, \quad \text{where } c_i \in \mathcal{V}$$

1. At prompt boundary $N$, the final hidden vector $h_N \in \mathbb{R}^{d}$ has already been projected through unembedding matrix $W_{\text{vocab}}$:
   $$z_N = h_N W_{\text{vocab}}^T \in \mathbb{R}^{|\mathcal{V}|}$$
2. The candidate logits are sliced directly from GPU buffer `buf_logits_`:
   $$z_{\text{sliced}}[i] = z_N[c_i], \quad \forall i \in \{0, \dots, K-1\}$$
3. Probability distribution is normalized over $\mathcal{C}$ with temperature scaling $T$:
   $$P(c_i) = \frac{\exp(z_{\text{sliced}}[i] / T)}{\sum_{j=0}^{K-1} \exp(z_{\text{sliced}}[j] / T)}$$
4. Decision is resolved instantly:
   $$i^* = \operatorname{argmax}_{i} (z_{\text{sliced}}[i]), \quad c^* = \mathcal{C}[i^*]$$

---

## 3. Implementation Blueprint

### Phase 1: Core Engine Integration (`src/server_single.cpp`)
1. **`ZeroDecodeResult` Data Structure**:
   - `int winning_index`
   - `std::string winning_choice`
   - `float confidence`
   - `std::vector<float> probabilities`
   - `std::map<std::string, float> choice_probabilities`
   - `double prefill_ms, total_ms`
   - `int decode_passes` (strictly 0)
2. **Token Mapping & Multi-Candidate Slicing**:
   - Helper to tokenize candidate choices with and without leading space prefixes (e.g. `" APPROVED"`, `"APPROVED"`, `" 0"`, `"0"`).
   - Resolve single-token representation or maximum prefix energy for each candidate.
   - Device-to-Host transfer of the sliced candidate logits ($K \times 4$ bytes) instead of transferring entire 248k vocabulary ($1\text{ MB}$).
3. **`MoeEngine::execute_zero_decode_bypass`**:
   - Reuses existing KV prefix cache / prefill pipeline.
   - Synchronizes `main_stream_` after prompt prefill.
   - Slices logits, computes softmax distribution, and assembles formatted output on CPU.

### Phase 2: HTTP Server & OpenAI Protocol Extension
1. **Dedicated Endpoint `/v1/decide`**:
   - Request parameters: `prompt` / `messages`, `choices` (`std::vector<std::string>`), `temperature`, `format` (`"json"`, `"text"`, `"object"`).
   - Responds with winning choice, confidence, calibrated probability distribution, and timing telemetry.
2. **OpenAI `/v1/chat/completions` Compatibility**:
   - Check for `zero_decode_bypass: true` flag and `choices: [...]` in request JSON.
   - When present, triggers ZDB fast-path and returns standard OpenAI-compatible `chat.completion` response format with `usage.completion_tokens = 0` and `telemetry.zero_decode_bypass = true`.

### Phase 3: Verification, Examples & Publication-Grade Benchmark Harness
1. **Example Scripts**:
   - `test_zdb_approval.ps1`: Binary loan/access approval routing.
   - `test_zdb_sentiment.ps1`: Multi-class 3-way sentiment classification.
   - `test_zdb_tool_router.ps1`: Fast 4-way agentic tool selection.
2. **Benchmarking Suite (`run_benchmark_zdb.ps1`)**:
   - Compare ZDB Enabled vs. Disabled (Traditional Autoregressive Decoding).
   - Metrics: End-to-end Latency (ms), Time to First Token (TTFT), Decode Passes, Memory Bandwidth Saved, Speedup Factor.
   - Validate numerical agreement between ZDB step-$N$ argmax and autoregressive step 1 greedy output.
