# Contributing to MinnieTheMoEcher

Thank you for your interest in contributing to **MinnieTheMoEcher**! Whether you are a human software engineer or an autonomous AI agent pair-programming on this repository, these guidelines ensure seamless collaboration, zero git merge conflicts, and 100% preservation of bare-metal inference performance.

> 💡 **Interactive Architecture Portal:**  
> For an interactive visual codebase explorer, sitemap tree, and architecture walkthrough, open [`DEVELOPER_GUIDE.html`](DEVELOPER_GUIDE.html) in your browser.  
> You can also view the raw architectural spec at [`DEVELOPER_GUIDE.md`](DEVELOPER_GUIDE.md).

---

## 1. Core Principles & Philosophy: Static-Linked Modularity

MinnieTheMoEcher is engineered for high-throughput, low-latency bare-metal inference on **NVIDIA GPUs (CUDA sm_80–sm_120)** and **Apple Silicon (Metal Performance Shaders/Primitives)**. Every CPU microsecond and VRAM transaction is critical.

To enable multiple developers and AI agents to work on separate features and model implementations in parallel without merge conflicts, the codebase strictly adheres to **Pure C++ Static Linking**:

1. **Zero IPC / Zero Microservices:** Everything compiles and statically links into a single executable (`moecher.exe` or `moecher`).
2. **Zero Runtime Indirection on Hot Paths:** No dynamic `.dll`/`.so` `dlopen` loading, no virtual method dispatch tables (`vtable`) in the token generation loop, and no heap allocations during decode passes.
3. **Component Ownership & File Isolation:** Subsystems live in independent translation units (`.cpp` / `.hpp`) or inline implementation files (`.inl`).

---

## 2. Multi-Agent & Multi-Developer Branching Matrix

To avoid merge collisions, always branch off the baseline (`main` or `refactor/static-linked-modularity`) and adhere strictly to component ownership:

| Subsystem | Primary Files to Touch | Branch Naming Convention | Merge Collision Risk |
| :--- | :--- | :--- | :--- |
| **New Model Architecture** | `src/models/model_<name>.inl` | `feature/model-<name>` | **0% (Completely Isolated)** |
| **Agentic / Tooling / MCP** | `src/agentic.hpp`, `src/agentic.cpp` | `feature/agentic-<name>` | **0% (Completely Isolated)** |
| **Multimodal Vision** | `src/multimodal.hpp`, `src/multimodal.cpp` | `feature/vision-<name>` | **0% (Completely Isolated)** |
| **Zero Decode Bypass (ZDB)** | `src/zdb.hpp`, `src/zdb.cpp` | `feature/zdb-<name>` | **0% (Completely Isolated)** |
| **Web UI & Streaming** | `src/web_ui.hpp`, `src/web_ui.cpp` | `feature/webui-<name>` | **0% (Completely Isolated)** |
| **CUDA Kernels** | `src/cuda/activations.cu`, `activations.cuh` | `perf/cuda-<kernel>` | Low (Isolated Translation Unit) |
| **Apple Metal Backend** | `src/metal/metal_backend.mm` | `feature/metal-<name>` | Low (Isolated Translation Unit) |

---

## 3. Codebase Sitemap

```
src/
├── main.cpp                     # CLI arg parsing, hardware probing, thread affinity, server bootstrap
├── engine.hpp                   # Core MoecherEngine orchestrator, memory management, KV cache lifecycle
├── common.hpp                   # Global definitions, ModelConfig, GpuCapabilities, GPUTensor, CUDA_CHECK
├── expert_loader.hpp            # Direct I/O NVMe offloading, L1 VRAM cache, L2 DRAM cache, DMA staging
├── speculative.hpp              # PLD drafter, Qwen 2B drafter, MTPSelfDrafter, Tree MTP topology
│
├── logging.hpp / logging.cpp    # Centralized logging (LOG_INFO, LOG_ERROR), global feature flags
├── web_ui.hpp / web_ui.cpp      # Embedded Web UI assets, HTML sandbox, /status, /metrics
├── agentic.hpp / agentic.cpp    # Tool schemas, server-side tool execution, HTTP forward proxy & PAC
├── tokenizer.hpp / tokenizer.cpp# HuggingFace BPE tokenizer, Unicode/CJK normalization, special tokens
├── multimodal.hpp / multimodal.cpp # ViT encoder pass, image preprocessing, FrankensTin visual projection
├── zdb.hpp / zdb.cpp            # Zero Decode Bypass logit slicing, candidate scoring, Hybrid Arbiter gating
├── server.hpp / server.cpp      # HTTP REST routes, SSE streaming deltas, pinned System KV cache snapshot
│
├── models/                      # ARCHITECTURE PASSES (Isolated for Git branches!)
│   ├── model_qwen.inl           # Qwen 3.8 27B: Gated DeltaNet Linear Attention + GQA + MTP drafting
│   ├── model_deepseek.inl       # DeepSeek V4-Flash: MLA low-rank attention + 11k-expert MoE forward
│   └── model_<name>.inl         # Place new model passes here (e.g. model_mistral4.inl)
│
├── cuda/
│   ├── activations.cu / .cuh    # DeltaNet SSM recurrence, fused SwiGLU, RMSNorm, RoPE kernels
│   └── vision_kernels.cu        # ViT patch extraction and spatial merge kernels
│
└── metal/
    ├── metal_backend.h
    └── metal_backend.mm         # Apple Silicon Metal backend (MPS + MPP tiled SRAM dequantization)
```

---

## 4. How to Add a New Model Architecture

Model forward passes are implemented as `.inl` member files inside `src/models/`. This enables full direct member pointer arithmetic and compiler inlining while keeping git branches completely disjoint.

### Step 1: Branch Out
```bash
git checkout -b feature/model-mistral4
```

### Step 2: Create `src/models/model_mistral4.inl`
```cpp
#pragma once

bool MoecherEngine::forward_mistral4(int token, int pos, bool is_prefill, int draft_idx) {
    // Implement layer loops, attention, and MLP passes using engine member buffers
    // e.g. d_intermediate_, cublas_handle_, main_stream_
    return true;
}
```

### Step 3: Register in `src/engine.hpp`
1. Declare the private member function in `class MoecherEngine`:
   ```cpp
   bool forward_mistral4(int token, int pos, bool is_prefill = false, int draft_idx = 0);
   ```
2. Dispatch inside `MoecherEngine::forward()`:
   ```cpp
   if (cfg_.architecture == "mistral4") {
       return forward_mistral4(token, pos, is_prefill, draft_idx);
   }
   ```
3. Include the implementation at the bottom of `src/engine.hpp`:
   ```cpp
   #include "models/model_mistral4.inl"
   ```

---

## 5. How to Add New Agentic Tools

Built-in tool calling, OpenAI tool schema injection, and MCP (Model Context Protocol) integration are completely decoupled in `src/agentic.hpp` and `src/agentic.cpp`.

1. Open `src/agentic.cpp` and append your tool JSON schema to `get_builtin_tools_json()`.
2. Add your execution handler in `execute_system_command()` or create a dedicated function:
   ```cpp
   std::string execute_custom_tool(const std::string& args_json);
   ```
3. Hook your function in `execute_tool_call()` in `src/agentic.cpp`.
4. Rebuild with `.\build_fast.bat`.

---

## 6. How to Modify the Web UI

All frontend interface assets (HTML, CSS, JS) and HTTP routes are managed in `src/web_ui.hpp` and `src/web_ui.cpp`.
* Frontend assets are embedded directly as raw string literals.
* Edit `src/web_ui.cpp` directly or use `python scripts/embed_web.py web/ src/web_ui.cpp`.
* Rebuild with `.\build_fast.bat` and test in your browser at `http://localhost:8001`.

---

## 7. How to Add or Modify Zero Decode Bypass (ZDB) Heuristics

All non-autoregressive logit slicing, candidate scoring, and Hybrid Arbiter gate logic reside in `src/zdb.hpp` and `src/zdb.cpp`:
* Logit slicing: `moecher::zdb::slice_candidate_logits()`
* Arbiter Gate Evaluation: `moecher::zdb::evaluate_hybrid_arbiter()`
* Fast-path routing: If `gate_passed == true`, the server returns instantly with `completion_tokens = 0`.

---

## 8. Build & Verification Checklist

### Compiling on Windows
* **Fast compilation (MSVC + NVCC):**
  ```powershell
  .\build_fast.bat
  ```
* **Production CMake:**
  ```powershell
  cmake -B build
  cmake --build build --config Release
  ```

### Compiling on Apple Silicon (macOS Metal)
```bash
cmake -B build -DUSE_METAL=ON
cmake --build build --config Release
```

### Mandatory 0% Loss Verification Battery
Before opening a Pull Request, start the server and run the verification battery:

```powershell
# 1. Start server in a background window
.\moecher.exe --manifest <path_to_manifest.json> --port 8001 --no-vision

# 2. Run ZDB & System 2 Underwriting Fallback Suite
powershell -ExecutionPolicy Bypass -File scripts\test_chat_completions.ps1

# 3. Run Hybrid Arbiter & /v1/decide Matrix
powershell -ExecutionPolicy Bypass -File scripts\verify_hybrid_arbiter.ps1
```

**Acceptance Criteria for PRs:**
- Exit code `0` on all verification scripts.
- Exactly 0 heap memory growth in decode passes.
- No changes to underlying tensor math without explicit benchmark validation.
