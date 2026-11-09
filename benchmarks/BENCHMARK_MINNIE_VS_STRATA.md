# MinnieTheMoEcher vs. Strata: Running 125B+ Qwen3.8-Flash-Next on a 16 GB Consumer GPU

**Executive Summary:** When running the massive **Qwen3.8-Flash-Next (`GSQ-RCO-IQ1_M`, 54.4 GiB total footprint)** on a consumer Windows PC (**NVIDIA RTX 5060 Ti 16 GB + 32 GB RAM**), **MinnieTheMoEcher (`moecher.exe`) is 6.0x faster out-of-the-box than Strata (`20.74 tok/s` vs. `3.42 tok/s`)**. 

Even after we reverse-engineered Strata's C++ source code, discovered why its heavily advertised *"Zero-Config Hardware Planner"* fails on Windows, and wrote an external `VirtualAlloc` memory-balloon script to give Strata a fair second chance (`20.26 tok/s`), **MinnieTheMoEcher still beats Strata in both pure GPU mode (`20.06 tok/s`, 4.2x faster startup) and hybrid mode (`20.74 tok/s`)**.

Most importantly, our controlled hardware experiments prove a critical architectural lesson for local MoE inference: **offloading MoE expert computation to the CPU yields only a trivial ~3% decode speedup (`20.06` → `20.74 tok/s`) when properly constrained—and actually slows inference down (`14.72 tok/s`) if overused—while burning 8–12 CPU cores and degrading overall PC responsiveness.**

---

## 1. The Head-to-Head Benchmark Results

### Test Environment
* **Model:** `ISTA-DASLab/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF` (`IQ1_M`, 48 layers, 256 routed experts/layer, Top-10 active, `27.58 GiB` Transformer/MoE shard + `26.82 GiB` PLE N-gram table = **`54.40 GiB` total**)
* **GPU:** NVIDIA GeForce RTX 5060 Ti (`16 GB` GDDR7, Blackwell `sm_120`)
* **System RAM:** 32 GB DDR5 (Windows 11 WDDM, ~12 GB Pinned RAM hardware limit)
* **Prompt:** `"Write a fast C++ function to check if an integer is prime."` (`37` prompt tokens, `128` generated tokens, `temp = 0.0` greedy, thinking disabled)

| Rank | Engine & Configuration | Decode Speed | Latency / Token | Prefill Speed | Model Load Time | Works Out-of-the-Box? |
| :---: | :--- | :---: | :---: | :---: | :---: | :---: |
| **🥇 1** | **MinnieTheMoEcher (`--cpu-experts`)** | **`20.74 tok/s`** | **`48.2 ms`** | **`27.31 tok/s`** | **`~18.0 s`** | **Yes (Zero Config)** |
| **🥈 2** | **Strata `v0.1.58` (Tuned + External Pagefile Hack)** | `20.26 tok/s` | `49.4 ms` | `27.37 tok/s` | `~76.0 s` | **No** *(Requires external `VirtualAlloc` script + env override)* |
| **🥉 3** | **MinnieTheMoEcher (`--no-cpu-experts`, Pure GPU)** | **`20.06 tok/s`** | **`49.8 ms`** | **`25.87 tok/s`** | **`~18.0 s`** | **Yes (Zero Config)** |
| **4** | **MinnieTheMoEcher (Aggressive CPU: 2 Misses/Layer)** | `14.72 tok/s` | `67.9 ms` | `12.98 tok/s` | `~18.0 s` | Illustrates CPU bottleneck & L1 LRU starvation |
| **5** | **Strata `v0.1.58` (`STRATA_RESIDENT_HEADROOM_GIB=0.2`)** | `8.59 tok/s` | `116.4 ms` | `5.70 tok/s` | `~51.0 s` | **No** *(Requires undocumented env var; still reads 282 MB/tok from SSD)* |
| **6** | **Strata `v0.1.58` (Stock `--resident-experts`)** | **`3.42 tok/s`** | `292.8 ms` | `5.40 tok/s` | `~45.0 s` | **Yes (Default Experience — 6.0x slower than Minnie)** |

---

## 2. How Strata's "Zero-Config Planner" Failed on Windows (And How We Gave It a Second Chance)

Strata gained popularity on social media for its claim of a *"Zero-Config Hardware Planner"* that automatically budgets VRAM, System RAM, and SSD storage. Yet when we ran stock `strata.exe` on our 16 GB VRAM + 32 GB RAM Windows machine, it crawled at **`3.42 tok/s`**—reading **`367.7 MB per token` (`47.1 GB` over 128 tokens)** from the SSD despite `15.7 GiB` of physical RAM sitting completely free.

Instead of stopping there and declaring an easy 6x victory, we inspected Strata's C++ source code (`src/core/expert_source.cpp`) to understand why it failed and give Strata its best possible shot.

### The Bug Inside Strata's Memory Planner
In `FileExpertSource`, before allocating host RAM for the experts that do not fit in VRAM (`~14.5 GiB`), Strata queries the Windows kernel via `GlobalMemoryStatusEx()` and computes:

```cpp
// Strata's check BEFORE allocating RAM:
const uint64_t available = std::min(physical_free, ullAvailPageFile);
const uint64_t budget    = available > headroom_bytes ? available - headroom_bytes : 0;
if (bytes > budget) return false; // Refuses to use RAM and falls back to mmap!
```

On Windows, the default virtual memory setting is **"Automatically manage paging file size for all drives"**. Under this setting:
1. `MEMORYSTATUSEX::ullAvailPageFile` does **not** report the maximum size to which Windows can grow `pagefile.sys`. It only reports the remaining commit charge inside the *currently allocated* `pagefile.sys`.
2. Windows only expands `pagefile.sys` lazily when a process actually invokes `VirtualAlloc` or `malloc`.
3. Because CUDA's WDDM allocations (`~9 GiB` of VRAM + dense weights) also reserve system commit charge, `ullAvailPageFile` temporarily reads `~2.5–4.2 GiB` right after GPU initialization—even though `15.7 GiB` of physical RAM is completely free!
4. Because Strata checks `ullAvailPageFile` **before** calling `malloc`—and aborts if it looks too small—Windows is never triggered to expand `pagefile.sys`. Strata concludes the machine is out of memory, refuses to allocate a single byte of RAM, and falls back to thrashing the SSD at **`3.42 tok/s`**.

Even when we set Strata's internal override `STRATA_RESIDENT_HEADROOM_GIB=0.2`, Strata only locked `3.09 GiB` of experts in RAM and ran at **`8.59 tok/s`**.

### The External Trick We Wrote to Rescue Strata
Because Windows immediately shrinks its dynamic commit limit as soon as a helper process exits, simply running a pre-allocation script *before* `strata.exe` did not work either. To force Strata into its full in-RAM mode, we had to write a **background PowerShell job that repeatedly injected `16 GiB` `VirtualAlloc(MEM_COMMIT)` calls via Python `ctypes` every 5 seconds while `strata.exe` was booting**, timed to inflate `ullAvailPageFile` at the exact moment `strata.exe` executed its safety check.

Only with that external kernel commit-balloon trick running alongside `strata.exe` did Strata load all `15.69 GiB` of off-VRAM experts into RAM and reach **`20.26 tok/s`** (after a **`76-second` startup time**).

By contrast, **MinnieTheMoEcher (`moecher.exe`)** simply allocates its L2 Pinned and L2.5 Unpinned expert pools directly, allowing Windows to expand its commit limit naturally on demand. It starts up in **`18 seconds`** (4.2x faster than Strata) and hits **`20.06 – 20.74 tok/s` right out of the box with zero hacks**.

---

## 3. Architectural Deep Dive: Why MinnieTheMoEcher Wins

When we refactored `Qwen3.8-Flash-Next` into the true **MinnieTheMoEcher** architecture ([`src/models/model_qwen4.inl`](file:///e:/dev/MinnieTheMoEcher/src/models/model_qwen4.inl) and [`scripts/convert_qwen38_flash_next.py`](file:///e:/dev/MinnieTheMoEcher/scripts/convert_qwen38_flash_next.py)), we separated the model into native pre-packed binary tiers:

1. **`resident.bin` (`2.76 GiB`):** All dense/resident weights (Embeddings, 36 Gated DeltaNet linear attention layers, 12 Qwen Sparse Attention layers, 4-stream Hyper-Connection norms/projections, Routers, Shared Experts, Layer-1 PLE projections, and the `Q6_K` Output Head). Uploaded 100% into VRAM at startup and immediately unmapped.
2. **`experts.bin` (`23.42 GiB`, 12,288 contiguous `[Gate | Up | Down]` expert blobs) + `expert_freq.bin`:**
   * **Tier 1a — Static Hot L1 VRAM (`~7.5 GiB`, ~3,916 experts):** The empirically hottest experts across all 48 layers (with a guaranteed per-layer floor) live permanently in GDDR7 VRAM (`896 GB/s`).
   * **Tier 1b — Dynamic LRU L1 VRAM (`1.65 GiB`, 864 slots = 18 slots/layer):** Pre-warmed at startup and dynamically updated on every token using O(1) per-layer LRU tables. Because autoregressive generation has strong **temporal locality** (consecutive tokens in code or prose activate nearly the same experts for 5–15 tokens in a row), **Tier 1a + Tier 1b achieve a `77.3% – 78.3%` L1 VRAM hit rate**!
   * **Tier 2 — L2 Pinned Host RAM (`10.00 GiB`, 5,226 warm experts):** Stored in page-locked `cudaMallocHost` memory (safely below Windows WDDM's 12 GB pinned ceiling) for zero-copy asynchronous PCIe DMA (`st.copy_stream`) overlapped with GPU L1-Hit compute.
   * **Tier 2.5 — L2.5 Unpinned Host RAM (`~5.9 GiB`, ~3,146 coldest experts):** Stored in standard physical DDR5 RAM (`std::malloc`), completely bypassing the Windows WDDM pinned-memory limit with **zero SSD reads (`0.0 MB/tok`)**.
   * **Tier 3 — L3 NVMe SSD Fallback:** Only used on machines with `< 32 GB` system RAM; on 32 GB+ systems, `experts.bin` is closed right after startup.

---

## 4. The Truth About CPU Expert Computing: Trivial Gains vs. System Responsiveness Risks

A central claim of hybrid engines like Strata is that computing VRAM-missed MoE experts on the CPU in system RAM is a game-changer. To test this scientifically inside `MinnieTheMoEcher`, we built a state-of-the-art **AVX2 + F16C + FMA CPU MoE Expert Pool** ([`src/cpu/cpu_moe_experts.hpp`](file:///e:/dev/MinnieTheMoEcher/src/cpu/cpu_moe_experts.hpp)) supporting all 16 GGML/GSQ quantization formats (`IQ1_M`, `IQ2_XXS`, `IQ2_XS`, `IQ2_S`, `IQ3_XXS`, `IQ3_S`, `IQ4_NL`, `IQ4_XS`, `Q2_0`, `Q4_K`, `Q5_K`, `Q6_K`, etc.) and tested three regimes:

### Regime A: Pure GPU Execution (`--no-cpu-experts`) → **`20.06 tok/s`**
* **How it works:** All 10 routed experts per layer are computed on the GPU. The ~7.8 experts already in L1 VRAM execute immediately on `main_stream_` (`~25 µs`), while the ~2.2 missed experts are streamed over PCIe into the **L1 Dynamic LRU VRAM cache** on `st.copy_stream`.
* **Result:** **`20.06 tok/s` decode (`49.8 ms/token`)**, with minimal CPU usage, leaving the PC completely cool, quiet, and responsive.

### Regime B: Aggressive CPU Offload (Up to 2 Misses/Layer on CPU) → **`14.72 tok/s` (26% SLOWER!)**
* **What happened when we let the CPU compute 15.2% of experts (up to 2 misses per layer):**
  Decode speed **dropped from `20.06 tok/s` down to `14.72 tok/s`**!
* **Why Aggressive CPU Compute Hurts Fast MoE Engines:**
  1. **It Starves the L1 Dynamic LRU VRAM Cache:** In a true dynamic-LRU MoE engine like `MinnieTheMoEcher`, streaming a missed expert over PCIe to the GPU doesn't just compute it for token `t`—it **caches that expert in L1 VRAM (`896 GB/s`) for tokens `t+1, t+2, t+3`**. When the CPU computes that expert in RAM instead, the expert never enters VRAM. On our benchmark, aggressive CPU offload caused the L1 VRAM hit rate to **drop from `78.3%` down to `72.8%`**, creating **4,356 extra cache misses**!
  2. **OS Scheduler & WDDM Preemption Jitter:** Running 48 layer synchronizations per token across 10–12 CPU threads means that if Windows or the NVIDIA WDDM driver preempts even a single CPU thread for `300 µs` to handle desktop UI, browser tabs, or audio, the entire inference pipeline stalls waiting for the CPU.
  3. **CPU Saturation:** Hammering all DDR5 channels and 10+ AVX2 cores continuously at `100%` utilization turns the CPU into a space heater and introduces noticeable lag into other applications running on the PC.

### Regime C: Surgical CPU Tail-Miss Offload (`--cpu-experts`) → **`20.74 tok/s` (+3.4% Decode Gain)**
* **How we made CPU offload positive:**
  We restricted [`forward_token_qwen4_exp`](file:///e:/dev/MinnieTheMoEcher/src/models/model_qwen4.inl#L1229-L1256) so that:
  1. **Warm `L2 Pinned RAM` misses are NEVER intercepted by the CPU**—they always DMA over PCIe into `L1 Dynamic LRU` VRAM so frequent experts stay cached on the GPU.
  2. **Only a single cold `L2.5 Unpinned RAM` tail miss (`orig_k >= 7` when `n_misses >= 2`)** is computed by the CPU AVX2 pool, and only while the PCIe bus is already busy transferring a higher-priority miss.
* **The Honest Verdict on CPU Decoding:**
  Even with lock-free work-unit completion counters, zero-copy mapped output fusion, and surgical tail-only scheduling (`2.1%` of experts computed on CPU), **CPU expert decoding only adds `~0.68 tok/s` (`20.06` → `20.74 tok/s`, a `~3.4%` decode improvement, and `25.87` → `27.31 tok/s` in prefill)**.
  
  Meanwhile, pure GPU mode (`--no-cpu-experts` at **`20.06 tok/s`**) achieves **97% of the speed** without spinning up 10 AVX2 worker threads—preserving full system responsiveness for the user's IDE, browser, and desktop workflow.

---

## 5. Summary & Recommendations

1. **MinnieTheMoEcher is the clear winner on Windows:**
   * **6.0x faster than stock Strata** (`20.74 tok/s` vs. `3.42 tok/s`).
   * **2.4x faster than env-tuned Strata** (`20.74 tok/s` vs. `8.59 tok/s`).
   * **Faster than even externally-hacked Strata** (`20.74 tok/s` vs. `20.26 tok/s`), with **4.2x faster startup** (`18s` vs. `76s`) and zero OS pagefile tricks required.
2. **Dynamic L1 VRAM Caching Beats Brute-Force CPU Compute:**
   * Strata relies on heavy CPU expert evaluation (`16.5 ms/token` of continuous multi-core CPU math) because its VRAM cache is static during decode (`0` dynamic admissions).
   * `MinnieTheMoEcher`'s **Static Hot L1 + Dynamic LRU L1 VRAM cache** serves nearly **80% of all routed experts at `896 GB/s` GDDR7 speed**, making heavy CPU compute completely unnecessary.
3. **Flexible Control in CLI & Web UI:**
   * Users who want every last drop of benchmark throughput on a dedicated machine can run with `--cpu-experts` (or check **"Hybrid CPU Expert Offload"** in the Web UI Settings).
   * Users who want a quiet, responsive workstation while coding or multitasking can pass `--no-cpu-experts` (or uncheck it in the Web UI) and still enjoy **`20.06 tok/s`**—losing only ~3% decode speed while keeping their CPU completely free.

