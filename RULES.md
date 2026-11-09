# MinnieTheMoEcher — Engineering & Agent Rules (`RULES.md`)

Whether you are a human software engineer or an autonomous AI coding agent working on **MinnieTheMoEcher**, you **MUST** follow all rules in this document at all times.

> 💡 **Interactive Architecture Portal:**  
> Open [`docs/DEVELOPER_GUIDE.html`](docs/DEVELOPER_GUIDE.html) in your browser or read [`docs/DEVELOPER_GUIDE.md`](docs/DEVELOPER_GUIDE.md) for the complete architecture walkthrough.

---

## Rule #1 — Zero User-Installed Dependencies & Zero Host OS Pollution
- The end user installs **only** the `moecher` executable (`moecher.exe` / `moecher`) and downloads model weights—period.
- **Never** require the end user to manually install Node.js, Python, npm packages, or system bloatware to run `moecher`.
- If a feature requires an external tool or runtime bridge (e.g., Hybrid Cloud CLI), `moecher` must automatically download, sandbox, and manage it strictly inside `.moecher_runtime/` with zero host OS pollution.

---

## Rule #2 — Clean Root Directory Policy
Keep the repository root strictly minimal. Only the following files belong in the project root:
- `README.md` — Main project overview and documentation index
- `RULES.md` — This file (authoritative rules for humans and AI agents)
- `LICENSE.txt` — Software license
- `CMakeLists.txt` — Root CMake build configuration
- `install.ps1` — Windows 1-click installer & builder
- `install.sh` — Linux (CUDA) & macOS (Apple Silicon Metal) 1-click installer & builder
- `build_fast.bat` — Fast incremental Windows developer build script
- `moecher.exe` / `moecher` — Compiled standalone binary (gitignored)

**Where everything else MUST go:**
- **Architectural Plans**: `docs/plans/`
- **Engineering Notes & Journals**: `docs/notes/`
- **User & Developer Guides / Manuals / Changelogs**: `docs/`
- **Published Articles & LinkedIn Drafts**: `Articles/`
- **Benchmark Reports, JSON Results & Benchmark Runners**: `benchmarks/` and `minniebench/`
- **Video, Slide & Audio Presentation Generators**: `presentations/`
- **Launchers, Model Downloaders & Conversion Scripts**: `scripts/` and `scripts/launchers/`
- **Unit & Manual Verification Tests**: `tests/` and `tests/manual/`
- **Temporary Scratch Files, Logs & Diffs**: `scratch/` (never commit scratch files or `.log` files)

---

## Rule #3 — Pure C++ Static-Linked Modularity & Zero Hot-Path Indirection
MinnieTheMoEcher is engineered for bare-metal inference on **NVIDIA GPUs (CUDA `sm_80`–`sm_120a`)** and **Apple Silicon (`Metal`)**:
1. **Zero IPC / Zero Microservices**: Everything compiles and statically links into a single executable (`moecher.exe` or `moecher`).
2. **Zero Runtime Indirection on Hot Paths**: No dynamic `.dll`/`.so` `dlopen` loading, no virtual method dispatch (`vtable`) inside token generation loops, and **zero heap allocations** during decode passes.
3. **Console Logging Quiet by Default**: Server and CLI inference run in quiet mode (`g_quiet = true`) by default so stdout I/O never bottlenecks token generation; use `--verbose` / `-v` when debugging.

---

## Rule #4 — Multi-Agent & Multi-Developer Component Isolation

| Subsystem | Primary Files to Touch | Branch Naming Convention | Merge Collision Risk |
| :--- | :--- | :--- | :--- |
| **New Model Architecture** | `src/models/model_<name>.inl` | `models/<name>` or `feature/model-<name>` | **0% (Completely Isolated)** |
| **Agentic / Tooling / MCP** | `src/agentic.hpp`, `src/agentic.cpp`, `src/tool_exec.hpp` | `feature/agentic-<name>` | **0% (Completely Isolated)** |
| **Multimodal Vision** | `src/multimodal.hpp`, `src/multimodal.cpp` | `feature/vision-<name>` | **0% (Completely Isolated)** |
| **Zero Decode Bypass (ZDB)** | `src/zdb.hpp`, `src/zdb.cpp` | `feature/zdb-<name>` | **0% (Completely Isolated)** |
| **Web UI & Streaming** | `web/index.html`, `web/script.js`, `web/style.css`, `src/web_ui.cpp` | `feature/webui-<name>` | **0% (Completely Isolated)** |
| **CUDA Kernels** | `src/cuda/*.cu`, `src/cuda/*.cuh` | `perf/cuda-<kernel>` | Low (Isolated Translation Unit) |
| **Apple Metal Backend** | `src/metal/metal_backend.mm` | `feature/metal-<name>` | Low (Isolated Translation Unit) |

---

## Rule #5 — Producer-Recommended Model Defaults & VRAM Safety
1. **Producer-Recommended Sampling & Reasoning Defaults**:
   - Every supported model family must automatically configure its producer-recommended sampling parameters (`temperature`, `top_p`, `top_k`, `min_p`, `repetition_penalty`, `reasoning_effort`, `thinking_budget`) in both C++ (`apply_sampling_defaults_*`) and the Web UI (`getModelRecommendedSampling`).
   - Do **not** hard-mask normal English tokens (such as `"I"` or `" I"`) in logit processors, and isolate repetition penalties so reasoning traces (`<think>...</think>`) do not penalize the final visible answer.
2. **Hardware-Aware Feature Scoping**:
   - Features tailored for a specific architecture (such as **Hybrid CPU Expert Offload**, which is designed exclusively for `Qwen 3.8 Flash-Next` / `QWEN4_EXP`) must be automatically disabled for other models.
   - `mtp_k` must default to `0` (auto: `1` on <=18GB `VRAM-Constrained` GPUs, `4` on >=24GB GPUs) so 16GB GPUs never run out of VRAM allocating speculative recurrent state slots.
3. **Web UI Security & Privacy**:
   - Never hardcode API keys or passwords in the codebase or `.moecher_config.json`.
   - Use `type="text" class="... masked-api-key" autocomplete="off" data-lpignore="true" data-1p-ignore="true"` for API key fields in `web/index.html` so browsers never autofill saved `localhost` passwords.

---

## Rule #6 — Quantization & Manifest Policy
When quantizing models or generating `moecher_manifest.json`:
1. **Embeddings & LM Head (`--no-embed-quant`)**:
   - For DeepSeek Q4/IQ2 manifests (`scripts/quantize_deepseek_dense.py`), **always** pass `--no-embed-quant` and `--no-indexer-quant`.
   - For ultra-sparse MoE models (`Qwen 3.8 Flash-Next` 512-expert MoE), keep shared experts, embeddings, LM head, router gates, and norms in high precision (`Q8_0` / `Q6_K` / `BF16`) and calibrate routed experts with an activation importance matrix (`imatrix`).
2. **Numerical Verification**:
   - Always verify bit-exact / cosine-similarity layer alignment and end-to-end prompt coherence before committing kernel or quantization changes.

---

## Rule #7 — Build & Verification Checklist

### Windows (CUDA + MSVC)
```powershell
# Fast incremental build (embeds web/ assets and updates moecher.exe in root & build/)
.\build_fast.bat

# Full CMake Release build (1-click installer)
powershell -ExecutionPolicy Bypass -File .\install.ps1
```

### Linux (CUDA) & macOS (Apple Silicon Metal)
```bash
# Unified 1-click installer & builder (auto-detects Linux CUDA or macOS Metal)
./install.sh
```

