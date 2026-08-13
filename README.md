# MinnieTheMoECher

Bare-metal C++/CUDA inference engine for **DeepSeek V4-Flash** (284B total / 13B active).

**Key feature**: O_DIRECT SSD offloading for MoE experts — runs the full unquantized model on a single RTX 3090 (24 GB).

## Prerequisites

- Linux (tested on Ubuntu 22.04/24.04)
- NVIDIA GPU with CUDA support (tested on RTX 3090 24GB and RTX 6000 Pro Blackwell 96GB)
- CUDA Toolkit 12.0+
- CMake 3.22+
- Python 3.10+ (for manifest generation and chat client)
- ~400 GB free disk space (for model weights)
- NVMe SSD strongly recommended (expert offloading throughput)

## Setup Guide

### Step 1: Download the Model

Download DeepSeek-V4-Flash from HuggingFace. You can use the `huggingface-cli`:

```bash
pip install huggingface_hub
huggingface-cli download deepseek-ai/DeepSeek-V4-Flash-0731
```

This downloads to `~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/` by default.

### Step 2: Generate Manifest and Binary Files

The build script reads the HuggingFace safetensors and creates three files:

| File | Description | Size |
|---|---|---|
| `moecher_manifest.json` | Tensor layout, offsets, and model config | ~280 KB |
| `attention_dense_layers.bin` | Resident weights (attention, norms, embeddings) | ~8 GB |
| `moe_experts.bin` | All routed expert weights, page-aligned for O_DIRECT | ~370 GB |

```bash
python3 scripts/build_manifest.py \
    --model-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731 \
    --output-dir .
```

> **Note:** This takes ~30 minutes and writes ~380 GB. The output directory needs sufficient free space.
> The manifest contains absolute paths to the generated `.bin` files and the HuggingFace `tokenizer.json`.

### Step 3: Build the Engine

```bash
mkdir -p build && cd build
cmake .. && make -j$(nproc)
cd ..
```

> **GPU Architecture:** The default CMakeLists.txt targets `sm_86` (Ampere/RTX 3090).
> For other GPUs, edit `CMAKE_CUDA_ARCHITECTURES` in `CMakeLists.txt`:
> - RTX 4090: `89`
> - RTX 5090 / Blackwell: `100`
> - Multiple targets: `"86;89;100"`

### Step 4: Run the Server

```bash
# Auto-detect VRAM
./build/moecher --manifest moecher_manifest.json --port 8001

# Or limit VRAM (e.g., 24 GB for RTX 3090 with other apps running)
./build/moecher --manifest moecher_manifest.json --port 8001 --max-vram 20
```

### Step 5: Chat

```bash
# Interactive multi-turn chat
python3 chat.py

# Or use curl (OpenAI-compatible API)
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"Hello!"}]}' | python3 -m json.tool
```

## Chat Client

The `chat.py` client supports:

| Command | Description |
|---|---|
| `/clear` | Clear conversation history |
| `/system <prompt>` | Set system prompt |
| `/temp <value>` | Set temperature (0.0–2.0) |
| `/tokens <n>` | Set max response tokens |
| `/help` | Show all commands |

No pip installs needed — uses only Python stdlib.

## Architecture

```
┌────────────────────────────────────────────┐
│  HTTP Server (cpp-httplib, port 8001)      │
│  OpenAI-compatible /v1/chat/completions    │
├────────────────────────────────────────────┤
│  Tokenizer (BPE from tokenizer.json)       │
├────────────────────────────────────────────┤
│  Forward Pass Engine                       │
│  ├─ Embedding                              │
│  ├─ 43x Transformer Layers                 │
│  │   ├─ HC Pre/Post (Hyper-Connections)    │
│  │   ├─ MLA Attention (absorbed low-rank)  │
│  │   │   ├─ Sliding window (128 tokens)    │
│  │   │   ├─ CSA/HCA compressed attention   │
│  │   │   └─ Attention sinks                │
│  │   └─ MoE FFN (256 experts, top-6)       │
│  └─ Head (HC reduce + logits)              │
├────────────────────────────────────────────┤
│  Expert Loader (O_DIRECT + LRU cache)      │
│  ├─ Page-aligned reads from NVMe           │
│  ├─ Async GPU prefetch via staging ring    │
│  └─ Auto-sizes cache to available VRAM     │
├────────────────────────────────────────────┤
│  CUDA Kernels (activations.cu)             │
│  ├─ RMSNorm, SiLU*mul, RoPE/YaRN          │
│  ├─ FP8/FP4 dequantization                 │
│  ├─ MLA attention (absorbed, inverse RoPE) │
│  └─ HC Sinkhorn, softmax, top-k            │
└────────────────────────────────────────────┘
```

## VRAM Usage

| Component | Size |
|-----------|------|
| Embedding + Head | ~2 GB |
| Attention weights (43 layers, FP8) | ~1.5 GB |
| Shared experts (43 layers, FP8) | ~2.2 GB |
| Gate + norms + HC params | ~1 GB |
| KV cache + working buffers | ~2 GB |
| **Total resident** | **~8.7 GB** |
| Expert cache (fills remaining VRAM) | auto |

- **RTX 3090 (24 GB):** ~14 GB expert cache → ~1000 experts cached out of 11,008 total
- **RTX 6000 Pro (96 GB):** All experts fit in VRAM → no SSD reads during inference

Use `--max-vram <GB>` to limit VRAM when running alongside other applications.

## Files

| File | Description |
|---|---|
| `scripts/build_manifest.py` | Generates manifest + binary files from HF safetensors |
| `src/server_single.cpp` | Main engine + HTTP server (~2700 lines) |
| `src/cuda/activations.cu` | Custom CUDA kernels |
| `src/cuda/activations.cuh` | Kernel declarations |
| `src/thread_pool.h` | Thread pool for async expert loading |
| `chat.py` | Interactive CLI chat client |
| `CMakeLists.txt` | Build system |
| `moecher_manifest.json` | Reference model config (regenerated per-machine by build_manifest.py) |

## License

Apache 2.0
