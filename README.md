# moecher

Bare-metal C++/CUDA inference engine for DeepSeek V4-Flash (284B total / 13B active).

**Key feature**: O_DIRECT SSD offloading for MoE experts — runs on 3090 24GB.

## Prerequisites

- NVIDIA GPU with CUDA support (tested on RTX 3090 and RTX 6000 Pro Blackwell)
- CUDA Toolkit 12.0+
- CMake 3.22+
- DeepSeek-V4-Flash model downloaded via HuggingFace

## Quick Start

```bash
# 1. Build the manifest and repack model weights (one-time, ~30 min)
python3 scripts/build_manifest.py \
    --model-dir ~/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash \
    --output-dir .

# 2. Build the engine
mkdir -p build && cd build
cmake .. && make -j$(nproc)

# 3. Start the server
./moecher --manifest ../moecher_manifest.json --port 8001

# 4. Test it
curl -X POST http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"What is the capital of France?"}]}'

# 5. Or use the interactive chat client
python3 chat.py
```

The chat client supports multi-turn conversations, slash commands (`/clear`, `/system`, `/temp`, `/tokens`), and colored terminal output. No pip installs needed.

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
│  │   ├─ MLA Attention (low-rank Q/O)       │
│  │   └─ MoE FFN (256 experts, top-6)       │
│  └─ Head (HC reduce + logits)              │
├────────────────────────────────────────────┤
│  Expert Loader (O_DIRECT + LRU cache)      │
│  ├─ Page-aligned reads from NVMe           │
│  ├─ LRU expert cache in VRAM               │
│  └─ Auto-sizes cache to available VRAM     │
├────────────────────────────────────────────┤
│  CUDA Kernels (activations.cu)             │
│  ├─ RMSNorm, SiLU*mul, RoPE/YaRN          │
│  ├─ FP8/FP4 dequantization                 │
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

On a 3090 (24 GB): ~14 GB expert cache → ~1000 experts cached out of 11,008 total.
On a 6000 Pro (96 GB): All experts fit in VRAM → no SSD reads during inference.

## Files

- `scripts/repack_aligned.py` — Repacks safetensors into page-aligned binary files + manifest
- `src/server_single.cpp` — Main engine + HTTP server
- `src/cuda/activations.cu` — Custom CUDA kernels
- `chat.py` — Interactive CLI chat client (multi-turn, slash commands)
- `CMakeLists.txt` — Build system
