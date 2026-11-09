# MinnieTheMoEcher: NVIDIA RTX 3090 Setup & Deployment Guide

This guide takes you step-by-step from cloning the repository to running the **DeepSeek-V4 MoE** model on a PC equipped with an **NVIDIA GeForce RTX 3090 (24 GB VRAM)**.

---

## ⚡ Quickstart (1-Click Automated Installer)

If you just want to get it running with zero hassle:
```bash
git clone https://github.com/tinobruno/minniethemoecher.git
cd minniethemoecher
./install.sh
```
The installer will automatically detect your RTX 3090, configure CUDA for `sm_86`, calculate optimal memory budgets, compile the engine, and create a `./start.sh` launcher.

---

## 1. System Requirements & Hardware Sizing

### Minimum Hardware
- **GPU**: NVIDIA GeForce RTX 3090 (24 GB GDDR6X, Compute Capability `sm_86`)
- **System RAM**: **64 GB** minimum (DDR4 or DDR5, dual/quad channel). **128 GB** recommended.
- **CPU**: Modern 8+ core x86_64 CPU (AMD Ryzen 5000/7000/9000 or Intel 12th+ Gen).
- **Storage**: Fast NVMe PCIe 4.0 SSD with at least **100 GB** free space.
- **Operating System**: Linux (Ubuntu 22.04 / 24.04 LTS or Arch / Fedora).

### Memory Architecture on RTX 3090 (24 GB VRAM + 64 GB DRAM)
MinnieTheMoEcher employs a high-throughput **Dual-Tier Dynamic Cache**:
```
┌────────────────────────────────────────────────────────────────────────┐
│                        24 GB GDDR6X VRAM                               │
│  ┌───────────────────────┬─────────────────┬─────────────────────────┐  │
│  │ Dense Layers (9.4 GB) │ KV Cache (4 GB) │ L1 Expert Cache (7.6GB) │  │
│  │ (Attention/Norm/Head) │ (Sliding Win)   │ (~1,150 IQ2_XXS Experts)│  │
│  └───────────────────────┴─────────────────┴─────────────────────────┘  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ Zero-Copy PCIe DMA (6.4+ GB/s)
┌───────────────────────────────────▼────────────────────────────────────┐
│                        64 GB SYSTEM RAM (DRAM)                         │
│  ┌───────────────────────────────────────────────────────────────────┐  │
│  │ L2 Pinned Expert Cache (45.0 - 55.0 GB, ~6,800 - 8,300 Experts)   │  │
│  └───────────────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────────────┘
```

---

## 2. Prerequisites & System Dependencies

### Install Linux Packages & CUDA Toolkit
Ensure you have **CUDA Toolkit >= 12.0** and build tools installed:

```bash
# Ubuntu / Debian
sudo apt-get update
sudo apt-get install -y build-essential cmake git curl \
                        python3 python3-pip python3-venv \
                        libcurl4-openssl-dev
```

Verify your NVIDIA driver and CUDA installation:
```bash
nvidia-smi
nvcc --version
```
*(Ensure `nvcc` is found and outputs CUDA 12.0 or higher)*.

---

## 3. Clone Repository & Build for RTX 3090 (`sm_86`)

### Step 3.1: Clone the Repository
```bash
git clone https://github.com/tinobruno/minniethemoecher.git
cd minniethemoecher
```

### Step 3.2: Configure & Compile with CUDA Architecture 86
The RTX 3090 uses NVIDIA Ampere architecture (`sm_86`). Pass `-DCMAKE_CUDA_ARCHITECTURES=86` to CMake:

```bash
# Create build directory
mkdir -p build && cd build

# Configure CMake targeting RTX 3090 (sm_86)
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CUDA_ARCHITECTURES=86

# Build all binaries in parallel
cmake --build . -j$(nproc)

# Return to project root
cd ..
```

This will produce the high-performance C++/CUDA server binary `./build/moecher`.

---

## 4. Preparing Model Weights & Manifest

You need the following model files in your project directory:

| File | Size | Description |
|---|---|---|
| `attention_dense_layers.bin` | ~9.44 GB | Dense attention, norms, and embeddings |
| `moe_experts_iq2.bin` | ~72.56 GiB | 2-bit IQ2_XXS + Q2_K MoE experts |
| `moecher_manifest_iq2.json` | ~280 KB | Model tensor offsets & architecture spec |
| `tokenizer.json` | ~1.5 MB | DeepSeek-V4 BPE Tokenizer |

### Option A: Use Pre-Quantized IQ2_XXS Expert Binary
If you already have `moe_experts_iq2.bin` and `attention_dense_layers.bin`, place them in the root directory:
```bash
ls -lh attention_dense_layers.bin moe_experts_iq2.bin moecher_manifest_iq2.json
```

### Option B: Quantize from Base Weights
If you have the base `moe_experts.bin`:
```bash
# Quantize all experts to 2-bit IQ2_XXS + Q2_K layout on RTX 3090:
python3 scripts/quantize_experts_iq2.py \
  --manifest moecher_manifest.json \
  --output-bin moe_experts_iq2.bin \
  --output-manifest moecher_manifest_iq2.json \
  --batch-size 16
```

### Option C: Extract from GGUF
If you have an official DeepSeek-V4 GGUF file:
```bash
python3 scripts/extract_gguf_experts.py \
  --gguf /path/to/model.gguf \
  --manifest-in moecher_manifest.json \
  --output-bin moe_experts_iq2.bin \
  --output-manifest moecher_manifest_iq2.json
```

---

## 5. Running the Engine on RTX 3090

Launch `./build/moecher` with memory flags optimized for the 24 GB VRAM + system RAM configuration:

```bash
./build/moecher \
  --manifest moecher_manifest_iq2.json \
  --max-vram 22 \
  --dram-cache-gb 48 \
  --quiet
```

### Explanation of Flags:
- `--manifest moecher_manifest_iq2.json`: Specifies the 2-bit model manifest.
- `--max-vram 22`: Limits total VRAM usage to **22.0 GB** (leaving 2.0 GB headroom for OS display and desktop composite buffers).
- `--dram-cache-gb 48`: Allocates **48.0 GB** of pinned host RAM as the L2 expert offload cache.
- `--quiet`: Suppresses raw per-token streaming in the server terminal for clean logging.
- `--port 8001`: Sets HTTP OpenAI-compatible server port (default: `8001`).

### What you should see on startup:
```
[INFO] ═══ moecher starting ═══
[INFO] Loading manifest: moecher_manifest_iq2.json
[INFO] Model: 43 layers, 256 experts, 6 active, hidden=4096, dtype=iq2_xxs
[INFO] VRAM: 23.9 GB free / 24.0 GB total
[INFO] Loading dense tensors from attention_dense_layers.bin
[INFO] Expert L1 (VRAM) cache budget: 7.6 GB
[INFO] Expert L2 (DRAM) cache budget: 48.0 GB
[INFO] Expert L1 cache: 1150 slots (7.6 GB)
[INFO] Expert L2 cache: 7280 slots (48.0 GB)
[INFO] Preloading 1150/11008 experts into VRAM...
[INFO] Preloading 7280/11008 experts into DRAM L2 cache...
[INFO] Server listening on port 8001
```

---

## 6. Interacting with the Model

### Method 1: Terminal Interactive Chat (`chat.py`)
In a new terminal window:
```bash
python3 chat.py
```

Single-prompt mode:
```bash
python3 chat.py --prompt "Explain the difference between L1 cache and L2 cache in modern CPUs."
```

### Method 2: Web Interface
Open your browser and navigate to:
```
http://localhost:8001/
```
The built-in web client includes live streaming, reasoning tree visualization, temperature/thinking budget toggles, and token-rate telemetry.

### Method 3: OpenAI-Compatible REST API
You can connect any OpenAI-compatible client (e.g. LangChain, LlamaIndex, Cursor, Continue.dev, LiteLLM):

```bash
curl http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "deepseek-v4-flash",
    "messages": [
      {"role": "system", "content": "You are a helpful assistant."},
      {"role": "user", "content": "Write a fast CUDA kernel for vector addition."}
    ],
    "temperature": 0.6,
    "max_tokens": 1024
  }'
```

---

## 7. Performance Tuning & BIOS Recommendations for RTX 3090

To achieve maximum offloading throughput (6.4+ GB/s DMA over PCIe):

1. **Enable Resizable BAR (ReBAR)** in your Motherboard BIOS:
   - Setting: `Above 4G Decoding` $\rightarrow$ **Enabled**
   - Setting: `Re-Size BAR Support` $\rightarrow$ **Enabled / Auto**
2. **PCIe Link Speed**:
   - Ensure the RTX 3090 is in the primary **PCIe 4.0 x16** slot directly connected to the CPU.
3. **RAM Speed**:
   - Enable **XMP / EXPO** profile in BIOS to ensure RAM operates at maximum memory clock (DDR4-3600 or DDR5-6000), providing up to 60–90 GB/s host memory bandwidth for expert swaps.
