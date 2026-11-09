# MinnieTheMoECher — Windows Native Guide (RTX 3090 / 4090 / 6000 Ada)

A high-performance, bare-metal C++/CUDA inference engine for **DeepSeek-V4-Flash MoE (34B active / 236B total)** running natively on Windows 10/11 x64.

---

## 🚀 Key Features on Windows

- **Zero WSL Overhead**: Runs natively on Windows with direct NVMe DMA streaming via Win32 `FILE_FLAG_NO_BUFFERING` and `CreateFileMappingA`.
- **100% Binary Weight Compatibility**: Uses the exact same binary weight files (`attention_dense_layers.bin`, `moe_experts_iq2.bin`) generated on Linux with zero re-quantization.
- **Built-in Web Chat UI**: Modern web interface with real-time token metrics, reasoning viewer, and live HTML inspector at `http://localhost:8001`.
- **1-Click Launch**: Double-click `start.bat` to launch.

---

## 🛠️ Prerequisites

Before building or running on Windows, make sure you have:

1. **NVIDIA GPU**: RTX 3080/3090 (24GB), RTX 4080/4090 (24GB), RTX 6000 Ada, or workstation GPU.
2. **System RAM**: $\ge 64\text{ GB}$ recommended for dual-tier DRAM expert caching.
3. **NVIDIA CUDA Toolkit 12.0+**: [Download from NVIDIA](https://developer.nvidia.com/cuda-downloads)
4. **Visual Studio 2022 / Build Tools** with the **"Desktop development with C++"** workload selected.
5. **CMake ($\ge 3.22$)**: [Download from CMake.org](https://cmake.org/download/) or run `winget install Kitware.CMake`.

---

## 📦 1-Click Automated Setup

### Step 1: Clone Repository
```powershell
git clone https://github.com/tinobruno/MinnieTheMoEcher.git
cd MinnieTheMoEcher
```

### Step 2: Build with PowerShell
Right-click `install.ps1` and select **Run with PowerShell**, or in terminal:
```powershell
powershell -ExecutionPolicy Bypass -File .\install.ps1
```
This automatically configures CMake, compiles `moecher.exe` with MSVC and CUDA, and prepares the workspace.

---

## 💾 Model Weights Setup

### Option 1: Qwen 3.8 27B INT4 (Native MTP Speculative Decoding ~98 tok/s)
Place the files from [TinoBruno/moecher-qwen-3.8-27b-q4](https://huggingface.co/TinoBruno/moecher-qwen-3.8-27b-q4):
- `attention_dense_layers_q4.bin` (~17.8 GB)
- `draft_vocab_ids.bin` (160 KB)
- `draft_lm_head_int8_bf16.bin` (390.6 MB)
- `moecher_manifest.json`

### Option 2: DeepSeek-V4-Flash (MoE 284B / 13B Active)
Place the weight files in the repository root:
- `attention_dense_layers.bin` (9.44 GB)
- `moe_experts_iq2.bin` (72.56 GB)
- `moecher_manifest_iq2.json`

*(Note: These files are 100% identical to the Linux version and can be copied directly from your Linux machine or SSD).*

---

## 🎯 Running MinnieTheMoECher

### Running Qwen 3.8 27B with MTP Speculative Decoding
```powershell
.\moecher.exe --manifest moecher_manifest.json --port 8001
```

### Running DeepSeek-V4-Flash MoE
```powershell
.\moecher.exe --manifest moecher_manifest_iq2.json --max-vram 85 --port 8001
```

### Consumer GPUs with DRAM Offloading (e.g. RTX 3090 / 4090 24GB)
```powershell
.\moecher.exe --manifest moecher_manifest_iq2.json --max-vram 22 --dram-cache-gb 48 --port 8001
```

### Command Line Arguments

| Parameter | Recommended (24GB GPU + 64GB RAM) | Description |
|---|---|---|
| `--manifest` | `moecher_manifest_iq2.json` | Path to the model manifest file |
| `--max-vram` | `22` | VRAM budget (in GB) dedicated to expert cache |
| `--dram-cache-gb` | `48` | Pinned host RAM cache (in GB) for sub-millisecond expert hits |
| `--port` | `8001` | Server HTTP port (default: 8001) |
| `--quiet` | *(flag)* | Suppress verbose token output and show clean generation telemetry |

---

## 🌐 Web Interface & API Access

- **Web Chat UI**: Open your browser at [http://localhost:8001/](http://localhost:8001/)
- **OpenAI Compatible Endpoint**: `http://localhost:8001/v1/chat/completions`

You can connect OpenWebUI, Continue.dev, Cursor, or any OpenAI-compatible client directly to `http://localhost:8001/v1`.
