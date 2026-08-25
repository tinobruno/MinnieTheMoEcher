# MinnieTheMoECher

Bare-metal C++/CUDA inference engine for **DeepSeek V4-Flash** (284B total / 13B active).

**Key features**:
- **Native 2-bit Vector Quantization (IQ2_XXS + Q2_K)**: All 11,008 MoE experts fit 100% in VRAM on a 96 GB GPU ($72.56\text{ GiB}$) with zero SSD reads during generation.
- **Full-Context Compressed Attention Cache (CSA & HCA)**: Dual-tier compressed KV memory spanning the entire 32K context window without token truncation or reasoning amnesia.
- **Fast BF16/FP16 Dense Prefill & MLA Attention**: High-throughput prompt processing and grouped low-rank MLA projection.
- **O_DIRECT SSD Offloading + Pinned DRAM L2 Cache**: Seamless fallback to stream experts from NVMe / DRAM on consumer GPUs (e.g. RTX 3090 24GB).
- **In-House Native Quantizer**: Directly quantizes base FP4 weights into calibrated `IQ2_XXS` + `Q2_K` format with optional activation importance matrix (`imatrix.dat`) without external GGUF tool dependencies.
- **Modern Web UI & Interactive Chat Client**: Full-featured Gemini-styled web interface with live reasoning traces, stop generation, and real-time TTFT/throughput telemetry.

---

## Prerequisites

- Linux (tested on Ubuntu 22.04 / 24.04)
- NVIDIA GPU with CUDA support (e.g. RTX 3090 24GB, RTX 4090 24GB, RTX PRO 6000 Blackwell 96GB)
- CUDA Toolkit 12.0+
- CMake 3.22+
- Python 3.10+ (for manifest generation, quantizer, and chat client)
- PyTorch 2.0+ (for quantizer scripts)
- NVMe SSD recommended for offloaded / partial VRAM setups

---

## Model Quantization Profiles

`MinnieTheMoEcher` supports four expert quantization profiles:

| Profile | Quantization Format | Expert Size | Total Model Size | VRAM Footprint | Recommended Use Case |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **Calibrated IQ2 (Recommended)** | `IQ2_XXS` ($w_1, w_3$) + `Q2_K` ($w_2$) | 7.08 MB | **72.56 GiB** | **100% Resident in 96GB VRAM** | Maximum reasoning accuracy & **~54 tok/s** speed on 96GB workstation GPUs |
| **Standard INT2** | Uniform 4-level scalar INT2 | 6.68 MB | **74.00 GiB** | **100% Resident in 96GB VRAM** | Uncalibrated fast baseline |
| **Imatrix INT2** | Column-weighted scalar INT2 | 6.68 MB | **74.00 GiB** | **100% Resident in 96GB VRAM** | Standard scalar with imatrix calibration |
| **Base FP4** | 4-bit floating point (E2M1) | 13.37 MB | **147.00 GiB** | Dynamic LRU Cache + NVMe Stream | Maximum weight fidelity |

---

## Performance & Benchmarks

Measured on **NVIDIA RTX PRO 6000 (Blackwell 96GB VRAM)**:

| Metric | Baseline Engine | Current Version (`moecher v2.05`) | Improvement |
| :--- | :--- | :--- | :--- |
| **Decode Throughput** | 35.2 tok/s | **53.98 tok/s** | **+53.4% faster** |
| **Time To First Token (TTFT)** | ~1.4 s | **694 – 817 ms** | **~2x faster** |
| **MoE VRAM Cache Coverage** | Partial NVMe offload | **100% in VRAM (11,008 experts)** | **Zero disk I/O stalls** |
| **MoE SwiGLU Kernel Launches** | 12 launches / layer | **1 fused launch / layer** | **12x launch reduction** |
| **Router Latency** | Host D2H sync | **Single GPU Block Reduction** | **Zero CPU-GPU sync stalls** |
| **Context Retention** | 32-token truncation | **Full 32K token reasoning cache** | **Zero reasoning context amnesia** |

---

## Setup Guide

### Step 1: Download DeepSeek-V4-Flash

Download the model checkpoint from HuggingFace:

```bash
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731
```

### Step 2: Build Base Manifest and Binary Weights

Extract attention layers, norms, embeddings, and base FP4 experts from safetensors:

```bash
python3 scripts/build_manifest.py \
    --model-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731 \
    --output-dir .
```

This creates:
- `moecher_manifest.json` (~280 KB)
- `attention_dense_layers.bin` (~8 GB resident weights)
- `moe_experts.bin` (~147 GB base FP4 expert weights)

---

### Step 3: Quantize Experts to IQ2_XXS + Q2_K (Recommended)

Run our native GPU-accelerated quantizer to convert base FP4 weights into calibrated `IQ2_XXS` + `Q2_K`:

```bash
# With imatrix calibration (recommended):
# 1. Generate imatrix interactively or via CLI:
python3 scripts/build_imatrix.py --prefix deepseek_v4_flash

# 2. Quantize experts with the generated imatrix:
python3 scripts/quantize_experts_iq2.py \
    --batch-size 32 \
    --imatrix imatrix/deepseek_v4_flash_imatrix.dat \
    --output-bin moe_experts_iq2.bin \
    --output-manifest moecher_manifest_iq2.json

# Or unweighted:
python3 scripts/quantize_experts_iq2.py --batch-size 32
```

---

### Step 4: Build the Engine

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..
```

> **GPU Architecture:** Default targets `sm_89;sm_90`. For other GPUs, configure `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`.

---

## Running the Server

### Server CLI Options
```bash
./build/moecher [OPTIONS]

Options:
  --manifest PATH      Path to model manifest JSON (required)
  --port PORT          HTTP server port (default: 8001)
  --host HOST          Server bind address (default: 0.0.0.0)
  --max-vram GB        VRAM budget cap in GB (e.g. 85 for 96GB GPU, 20 for 24GB GPU)
  --dram-cache-gb GB   Pinned DRAM L2 expert cache budget in GB (default: 0)
  --quiet              Suppress per-step debug logs
  --help               Display help message
```

### Execution Examples

#### 1. Calibrated IQ2_XXS (All Experts Resident in 96GB VRAM - Recommended)
```bash
./build/moecher --manifest moecher_manifest_iq2.json --max-vram 85 --quiet
```

#### 2. Consumer GPUs with NVMe + DRAM Offloading (e.g. RTX 3090 / 4090 24GB)
```bash
./build/moecher --manifest moecher_manifest_iq2.json --max-vram 20 --dram-cache-gb 24 --port 8001
```

#### 3. Base FP4 (Dynamic NVMe Streaming)
```bash
./build/moecher --manifest moecher_manifest.json --max-vram 80 --port 8001
```

---

## User Interfaces

### 1. Web UI (Gemini-Themed Modern Chat & Live HTML Workbench)

Moecher comes with a built-in modern web interface with rich reasoning telemetry and a dedicated HTML testing workbench:
- Open [`web/index.html`](file:///home/tinobruno/minniethemoecher/web/index.html) in your browser (or host via `python3 -m http.server 8080 --directory web`).
- **Features**:
  - **Resizable HTML Preview & Test Panel**:
    - **Live Sandboxed Execution**: Run interactive HTML, CSS, JavaScript, WebGL, Canvas, and SVG applications generated by Moecher in a safe sandbox.
    - **One-Click "Preview in Panel" Actions**: Every generated code block (and detected raw HTML documents) features an instant "Preview in Panel" trigger.
    - **Interactive Code Editor Tab**: Built-in editor with line numbers, code size indicator, Tab-indentation support, and live **Run Code** and **Format** actions.
    - **Integrated Console Inspector Tab**: Live interception of child `console.log`, `console.warn`, `console.error`, and uncaught JavaScript exceptions with level filters and error counter badges.
    - **Device Viewport Emulation**: Quick toggle between **Desktop (100%)**, **Tablet (768px)**, and **Mobile (375px)** device mockups.
    - **Testing Utilities**: Canvas background theme switcher (Dark / Light / Checkerboard for transparency/contrast testing), pop-out to new window via Blob URL, and one-click `.html` export/download.
    - **Draggable Split Resizer**: Custom panel sizing with memory persistence and fullscreen toggle.
  - **Live Stop Generation**: Abort in-flight reasoning and decoding instantly.
  - **Collapsible Reasoning**: Real-time thinking traces with token duration metrics.
  - **Performance Dashboard**: Real-time TTFT, prefill speed (t/s), decode speed (t/s), and turn counters.
  - **Tunable Generation Settings**: System prompt, temperature slider, max tokens, thinking token budget, and reasoning effort presets (`none`, `low`, `medium`, `high`, `xhigh`, `max`).
  - **Markdown & Code Rendering**: Syntax highlighted code blocks with one-click clipboard copying and live preview actions.

### 2. Interactive CLI Client (`chat.py`)

Connect to the server via terminal:

```bash
# Standard interactive chat:
python3 chat.py

# Display reasoning trace in terminal:
python3 chat.py --show-reasoning --temperature 0.6

# Custom budget and parameters:
python3 chat.py --url http://localhost:8001 --max-tokens 2048 --temperature 0.6 --thinking-budget 8192
```

### 3. OpenAI-Compatible API

```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "You are DeepSeek V4 Flash running locally."},
      {"role": "user", "content": "Write a self-contained HTML file with Three.js rendering a rotating cube."}
    ],
    "temperature": 0.6,
    "max_tokens": 4096,
    "thinking": {
      "type": "enabled",
      "budget_tokens": 4096
    }
  }' | jq .
```

---

## Architecture

```
┌────────────────────────────────────────────────────────────┐
│  HTTP Server (cpp-httplib, port 8001)                      │
│  OpenAI-compatible /v1/chat/completions & Web UI           │
├────────────────────────────────────────────────────────────┤
│  Tokenizer (BPE from tokenizer.json)                       │
├────────────────────────────────────────────────────────────┤
│  Dense Forward Pass Engine (BF16)                          │
│  ├─ Token Embedding & Unweighted RMSNorm                   │
│  ├─ 43x Transformer Layers                                 │
│  │   ├─ Hyper-Connections (Pre/Post Weighted Update)       │
│  │   ├─ Grouped Low-Rank MLA Attention                     │
│  │   ├─ Full-Context Compressed Attention (CSA & HCA)      │
│  │   └─ Top-6 Routed MoE (256 experts per layer)           │
│  └─ Final RMSNorm & Head Logits Generation                 │
├────────────────────────────────────────────────────────────┤
│  MoE Dispatch & Quantization Engine                        │
│  ├─ Fused IQ2_XXS SwiGLU (w1 Gate & w3 Up, in-register)    │
│  ├─ Q2_K GEMV (w2 Down, 16x16 Nested Sub-Block Quant)      │
│  ├─ GPU Top-6 Routing Reduction (Zero D2H stalls)          │
│  └─ Asynchronous Multi-Stream Pipeline (Q/KV & MoE overlap)│
├────────────────────────────────────────────────────────────┤
│  Memory & Cache Hierarchy                                  │
│  ├─ Full-Context Compressed KV Cache (up to 32K context)   │
│  ├─ L1 VRAM Cache (100% resident for IQ2 in 96GB)          │
│  ├─ L2 Pinned DRAM Cache (configurable --dram-cache-gb)    │
│  └─ Async O_DIRECT NVMe Prefetch Pipeline                  │
└────────────────────────────────────────────────────────────┘
```

---

## Changelog

### v2.06 — Live HTML Test & Preview Panel for Web UI
- **Resizable HTML Preview & Testing Workbench**: Added a high-performance 3-pane layout with a draggable resizer handle, sandbox iframe runner, built-in code editor, and live console inspector.
- **Universal Preview Actions & Smart Detection**: Every code block and HTML response generated by Moecher can be tested and interacted with in 1 click.
- **Device Emulation & Testing Utilities**: Desktop (100%), Tablet (768px), and Mobile (375px) responsive frames, background themes, Blob pop-outs, and one-click file download.

### v2.05 — Full-Context Compressed Cache & Attention Precision
- **Full-Context Compressed KV Cache**: Fixed 32-entry capacity capping bug in compressor cache, scaling storage across the full 32K token sequence context (`max_seq_len / ratio`). Eliminates reasoning context amnesia during long deliberation and code drafting.
- **Attention Softmax Scale Correction**: Removed double-scaling bug on attention dot products, aligning attention scaling strictly with $1 / \sqrt{d_{\text{head}}}$ matching CUDA reference kernels.
- **RoPE YaRN Unified Interpolation**: Refactored pair-coordinate RoPE YaRN ramp computation in `activations.cu` with unit magnitude scaling ($mscale = 1.0$), eliminating $(1.277)^2$ over-scaling.
- **Thinking & Deliberation Pipeline**: Implemented graceful `<think>` to `</think>` state transitions and handled early EOS gracefully during deliberation.
- **$O(V)$ Full-Vocabulary Min-P Sampler**: Streamlined vocabulary sampling with an efficient zero-sorting Min-P pass.
- **Multi-Turn Chat Formatting**: Added `<｜end▁of▁sentence｜>` delimiters between multi-turn assistant messages.

### v2.04 — Web UI & Interactive Controls
- **Gemini-Themed Web UI**: Added full web chat interface with real-time streaming, collapsible thinking traces, and latency/speed telemetry.
- **Live Stop Generation**: Added stream abort support via `AbortController`.
- **Reasoning Effort & Thinking Budget**: Added interactive controls for reasoning presets and token budgeting.

### v2.03 — 54 tok/s High-Performance Milestone
- **~54 tok/s Decode Throughput**: Boosted generation throughput from $35.2\text{ tok/s}$ baseline to **$53.98\text{ tok/s}$** on NVIDIA RTX PRO 6000 (Blackwell 96GB).
- **Asynchronous Multi-Stream Concurrency Engine**: Implemented non-blocking concurrency between Q and KV projections, and between Routed and Shared experts.
- **Vectorized 128-Bit Hyper-Connection (HC) Kernels**: 128-bit memory operations (`uint4` / `float4`) for hyper-connection pre/post kernels.
- **Fused SwiGLU MoE Execution**: Fused `w1` (gate) and `w3` (up) GEMV into a single CUDA kernel.
- **GPU-Accelerated Top-6 Routing**: Replaced CPU `std::partial_sort` with a GPU block reduction kernel.
- **100% VRAM Resident IQ2_XXS + Q2_K Experts**: Enabled all 11,008 MoE experts ($72.56\text{ GiB}$) to reside fully in 96GB VRAM.

---

## License

Apache 2.0
