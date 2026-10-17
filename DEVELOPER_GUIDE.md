# MinnieTheMoEcher: Developer & AI Agent Architecture Guide

> **Welcome to MinnieTheMoEcher.**
> This guide is written for software engineers and autonomous AI agents pair-programming on this codebase. It documents the directory structure, design principles, branching strategies, and exact step-by-step procedures to add new model architectures, agentic tools, multimodal capabilities, and UI features.

---

## 1. Architectural Philosophy: Static-Linked Modularity

MinnieTheMoEcher is engineered for bare-metal high-throughput inference (CUDA sm_80–sm_120 and Apple Silicon Metal). Every CPU microsecond and GPU memory transaction is critical.

To enable multiple developers and AI agents to work on separate features and model implementations in parallel without merge conflicts, the codebase uses **Pure C++ Static Linking**:

* **Zero IPC / Zero Microservices:** Everything compiles and statically links into a single executable (`moecher.exe` or `moecher`).
* **Zero Runtime Indirection on Hot Paths:** No dynamic `.dll`/`.so` `dlopen` loading, no virtual method dispatch tables (`vtable`) in the token generation loop, and no heap allocations during decode passes.
* **Separation of Concerns:** Translation units (`.cpp` / `.hpp`) and inline implementation files (`.inl`) isolate logical subsystems into distinct files.

---

## 2. Codebase Sitemap

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

## 3. Why `.inl` for Model Passes?

You will notice files in `src/models/` have the `.inl` extension (e.g. `model_qwen.inl`, `model_deepseek.inl`):

* **What it means:** `.inl` stands for **Inline Implementation**. These files contain C++ member function implementations of `class MoecherEngine`.
* **Why this is used:**
  1. **Direct Pointer Arithmetic:** The model forward pass accesses private `MoecherEngine` members (`d_qwen_qkv_`, `cublas_handle_`, `main_stream_`, KV caches) directly with zero accessor overhead.
  2. **Aggressive Compiler Optimization:** The MSVC and NVCC compilers inline loops and allocate registers across the entire forward pass as if written in a single file.
  3. **Zero Git Merge Conflicts:** Because each model architecture lives in its own `.inl` file, a developer or agent working on Qwen touches **only** `src/models/model_qwen.inl`. A developer working on DeepSeek touches **only** `src/models/model_deepseek.inl`.

---

## 4. How to Add a New Model Architecture

To implement a new model architecture (for example, `Mistral 4` or `Qwen Flash`):

### Step 1: Create a feature branch
```bash
git checkout -b feature/model-mistral4
```

### Step 2: Create the implementation file
Create `src/models/model_mistral4.inl`:
```cpp
#pragma once

// Forward declaration of member methods for Mistral 4 forward pass
bool MoecherEngine::forward_mistral4(int token, int pos, bool is_prefill, int draft_idx) {
    // Implement layer loops, attention, and MLP passes using engine member buffers
    // e.g., d_intermediate_, cublas_handle_, main_stream_
    return true;
}
```

### Step 3: Register in `src/engine.hpp`
1. Declare the method inside `class MoecherEngine`:
   ```cpp
   bool forward_mistral4(int token, int pos, bool is_prefill = false, int draft_idx = 0);
   ```
2. Include the implementation at the bottom of `src/engine.hpp`:
   ```cpp
   #include "models/model_mistral4.inl"
   ```
3. Add dispatch logic in `MoecherEngine::forward()`:
   ```cpp
   if (cfg_.architecture == "mistral4") {
       return forward_mistral4(token, pos, is_prefill, draft_idx);
   }
   ```

### Step 4: Build and test
```bash
.\build_fast.bat
.\moecher.exe --manifest models/mistral4/moecher_manifest.json --port 8001
```

---

## 5. How to Add New Agentic Tools

Server-side tool execution, OpenAI tool schema injection, and MCP (Model Context Protocol) integration are completely decoupled in `src/agentic.hpp` and `src/agentic.cpp`.

### To Add a Built-In Tool:
1. Open `src/agentic.cpp`.
2. Add your JSON schema definition to `get_builtin_tools_json()`.
3. Add the execution handler in `execute_system_command()` or create a dedicated function:
   ```cpp
   std::string my_new_tool(const std::string& input_arg);
   ```
4. Expose the function in `src/agentic.hpp` if needed outside the translation unit.
5. Rebuild with `.\build_fast.bat`.

---

## 6. How to Modify the Web UI

The Web UI assets are statically delivered by `src/web_ui.hpp` and `src/web_ui.cpp`.

* **Static Assets:** The HTML, CSS, and client-side JavaScript are embedded directly as raw string literals.
* **Editing the UI:** Edit `src/web_ui.cpp` directly or use `scripts/embed_web.py` to compile files from a frontend development folder.
* **Testing:** Rebuild with `.\build_fast.bat` and refresh `http://localhost:8001`.

---

## 7. How to Add or Modify Zero Decode Bypass (ZDB) Heuristics

All non-autoregressive logit slicing, candidate scoring, and Hybrid Arbiter gate logic reside in `src/zdb.hpp` and `src/zdb.cpp`:

* **Logit slicing:** `moecher::zdb::slice_candidate_logits()`
* **Arbiter Gate Evaluation:** `moecher::zdb::evaluate_hybrid_arbiter()`
* **System 1 vs System 2 Routing:** `src/server.cpp` calls `evaluate_hybrid_arbiter()`. If `gate_passed == true`, the server returns instantly without running the autoregressive decode loop.

---

## 8. Branching & Multi-Agent Collaboration Strategy

When assigning tasks to developers or AI agents, enforce strict component ownership:

| Feature Area | Primary Files Modified | Branch Naming Convention |
| :--- | :--- | :--- |
| **New Model Pass** | `src/models/model_<name>.inl` | `feature/model-<name>` |
| **Agentic / Tooling / MCP** | `src/agentic.hpp`, `src/agentic.cpp` | `feature/agentic-<name>` |
| **Multimodal / Vision** | `src/multimodal.hpp`, `src/multimodal.cpp` | `feature/vision-<name>` |
| **Zero Decode Bypass** | `src/zdb.hpp`, `src/zdb.cpp` | `feature/zdb-<name>` |
| **Web Interface** | `src/web_ui.hpp`, `src/web_ui.cpp` | `feature/webui-<name>` |
| **CUDA Kernels** | `src/cuda/activations.cu`, `activations.cuh` | `perf/cuda-<kernel>` |
| **Apple Metal Backend** | `src/metal/metal_backend.mm` | `feature/metal-<name>` |

Because each domain maps to isolated compilation units, branches can be developed and merged into `main` with **zero merge conflicts**.

---

## 9. Compilation & Verification Checklist

### Compiling on Windows (MSVC + NVCC)
* Fast compilation via batch script:
  ```powershell
  .\build_fast.bat
  ```
* Production build via CMake:
  ```powershell
  cmake -B build
  cmake --build build --config Release
  ```

### Compiling on Apple Silicon (macOS)
```bash
cmake -B build -DUSE_METAL=ON
cmake --build build --config Release
```

---

## 8. Tooling Safety & Sandboxing Architecture

MinnieTheMoEcher implements a strict 3-pillar defense-in-depth safety architecture to prevent unintended system execution or data leakage:

1. **Secure-by-Default Runtime:**
   - Server-side OS shell execution (`g_server_exec`), tool schemas (`g_enable_tools`), and file writes (`g_allow_file_write`) are disabled by default.
   - Standard pure inference and client-side tool calling (OpenAI standard `choices[0].message.tool_calls`) remain fully functional.
   - Server-side execution requires explicit opt-in via `--allow-host-exec` and `--allow-file-write`, which outputs a high-visibility security warning banner at startup.

2. **Compile-Time Host Gating:**
   - Standard release builds compile with `-DENABLE_UNSAFE_HOST_EXEC=OFF` (the CMake default).
   - In this mode, `execute_system_command()` is physically inert in the compiled binary.
   - Compiling a binary with host execution capability requires explicitly passing `-DENABLE_UNSAFE_HOST_EXEC=ON` to CMake.

3. **Strict Workspace Path Sandboxing:**
   - All built-in file operations (`read_file`, `write_file`, `edit_file`) are sandboxed within `--workspace <path>` (default: `./workspace`).
   - Directory traversal attempts (`../`) and absolute paths outside the workspace are blocked.
   - Access to sensitive files and credentials (`.env`, `.git`, `.ssh`, `id_rsa`, `cookie_jar.json`, `authorized_keys`) is rejected.
   - File modification requires runtime opt-in via `--allow-file-write`.

### 0% Loss Verification Battery
Before submitting a PR or merging into `main`, run the test suite to verify 0% regression:

1. **REST API & ZDB Decision Test:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\test_chat_completions.ps1
   ```
2. **Hybrid Arbiter Full Matrix:**
   ```powershell
   powershell -ExecutionPolicy Bypass -File scripts\verify_hybrid_arbiter.ps1
   ```
3. **Nemotron 3.5 SSM & MoE Architecture Unit Test:**
   ```powershell
   cmd.exe /c "scripts\build_test_nemotron.bat"
   ```
4. **Verify Zero Memory Growth:**
   Confirm that GPU VRAM usage and host memory working set remain constant during decode loops.

---

## 9. Nemotron 3.5 Lightning & FrankensTin Multimodal Integration

Nemotron 3.5 Lightning (30B-A3B) is MinnieTheMoEcher's ultra-fast agentic engine designed to fit 100% inside consumer <16GB VRAM GPUs (e.g. GeForce RTX 5060 Ti 16GB) and 24GB Apple Silicon Unified Memory:

* **Hybrid Sequence Mixing (`src/models/model_nemotron.inl`):**
  * Interleaves Mamba-2 State-Space Model (SSM) recurrence layers (`deltanet_ssm_batch_kernel_warp_fast`) with selective Grouped-Query Attention (GQA) layers.
  * Mamba-2 SSM layers maintain a constant-size recurrent state matrix ($d_{state}=128$), eliminating $O(N^2)$ KV-cache explosion and dramatically accelerating prompt prefill on 100k+ token codebases.
* **Sparse 3B Active MoE Execution:**
  * Out of 30B total parameters, only 3B are activated per token via top-K routing, reducing memory reads from 13.2 GB down to ~1.4 GB per token.
  * Achieves **~85 to 100 tok/s** on an RTX 5060 Ti 16GB.
* **Decoupled FrankensTin Multimodal Vision:**
  * Uses the decoupled FrankensTin vision delegation pipeline (`TinoBruno/frankenstin-vision-delegate`).
  * Image queries are lazily intercepted in `src/server.cpp`, processed through the 3.3 GB ViT delegate, and injected as high-fidelity `<visual_perception>` scene representations into the context stream.

