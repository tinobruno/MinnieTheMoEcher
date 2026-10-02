# MinnieTheMoEcher on Apple Silicon (Metal Backend)

MinnieTheMoEcher features a native Apple Metal compute backend engineered specifically for **Apple Silicon (M-series / Apple M6, macOS 15+)**. It provides bare-metal, zero-copy inference for 27B-parameter models with industry-leading prefill and decode speeds.

---

## Key Metal Features

1. **Hardware Metal Performance Primitives (MPP) Tiled GEMM Engine**:
   - On-the-fly weight dequantization directly inside 4 KB threadgroup on-chip SRAM (`threadgroup bfloat s_w[64][32]`).
   - Powered by Apple's native `<metal_tensor>` and `<MetalPerformancePrimitives/MetalPerformancePrimitives.h>` (`mpp::tensor_ops::matmul2d`).
   - Completely eliminates the 48 GB DRAM dequantization traffic bottleneck, delivering **143.3 to 172.0 tok/s prefill throughput** (2.67x faster than Apple MLX).
2. **Zero-Copy Unified Memory Architecture (UMA)**:
   - Allocations use `MTLResourceStorageModeShared`. Memory is shared coherently between the CPU host runtime and the GPU compute pipelines with zero PCIe transfers, zero D2D mirrors, and zero CPU/GPU copies.
3. **95.1% Memory Bus Saturation on Single-Token Decode**:
   - Streams quantized INT3 and INT4 weights at **146.0 GB/s sustained bandwidth** out of the 153.6 GB/s physical ceiling on 128-bit LPDDR5X-9600 memory buses.
   - Sustains **11.27 to 11.38 tok/s decode speed**, outperforming both `llama.cpp` (10.14 tok/s) and Apple MLX (9.81 tok/s).
4. **Fused DeltaNet SSM Recurrence & SIMD Projections**:
   - 48 DeltaNet linear attention layers execute with fused threadgroup recurrence and hardware SIMD horizontal reduction (`simd_sum`), reducing DeltaNet projection overhead from 1.75 ms to 0.178 ms per layer (9.8x speedup).
5. **Fused SwiGLU Activation Pipeline**:
   - Evaluates Gate and Up projections sequentially via MPP tensor units, barrier-synchronized with `[enc memoryBarrierWithScope:MTLBarrierScopeBuffers]`, and vectorized into `silu_mul_kernel`.
6. **Built-in OpenAI-Compatible HTTP Server & Web Workbench**:
   - Native streaming JSON server with KV cache prefix caching, multi-turn conversation memory, and live token telemetry.

---

## Hardware Requirements

- **Apple Silicon Mac**: Mac mini, MacBook Pro, Mac Studio, or iMac (Apple M1 through M6, Pro/Max/Ultra).
- **RAM**:
  - **16 GB RAM**: Supports quantized models up to 14B parameters.
  - **24 GB+ RAM** (Recommended): Runs Qwen 3.8 27B Vision (13.0 GB footprint) with zero memory pressure and zero swapping.
- **Operating System**: macOS 15.0+ (Sequoia or later).
- **Developer Tools**: Apple Xcode Command Line Tools (`clang`, `metal`).
- **CMake**: Version 3.22+ (`pip install cmake`).

---

## Quickstart & Build Instructions

### 1. Install Build Dependencies
If you have not already installed Command Line Tools:
```bash
xcode-select --install
```

Install CMake and Python dependencies:
```bash
python3 -m pip install cmake httplib requests
```

### 2. Compile the Metal Binary
Clone the repository and build using CMake:
```bash
git checkout porting/metal
cmake -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```
This builds two primary executables:
- `build/moecher`: The high-performance inference engine and server.
- `build/test_metal`: The comprehensive 12-stage Metal kernel unit test suite.

### 3. Verify Metal Hardware Tests
Run the Metal test suite to verify kernel numerical accuracy and device detection:
```bash
./build/test_metal
```
Expected output:
```text
==========================================================
  MinnieTheMoEcher — Metal Backend Test Suite (Apple M6)  
==========================================================
[TEST] 1. Device detection and unified memory... PASS
[TEST] 2. RMSNorm Metal compute kernel... PASS
[TEST] 3. Softmax & Argmax Metal compute kernel... PASS
[TEST] 4. Accelerate BLAS GEMM (cublasGemmEx)... PASS
[TEST] 5. Fused SwiGLU / silu_mul Metal compute kernel... PASS
[TEST] 6. RoPE Metal compute kernel... PASS
[TEST] 7. INT4 GEMV f32 (LM Head)... PASS
[TEST] 8. INT3 GEMV & Dequantization... PASS
[TEST] 9. DeltaNet Linear Attention recurrence... PASS
[TEST] 10. INT4 GEMV Metal GPU kernel (gemv_int4_cuda)... PASS
[TEST] 11. INT3 GEMV Metal GPU kernel (gemv_int3_cuda)... PASS
[TEST] 12. GQA Attention Metal compute kernel... PASS
==========================================================
  ALL TESTS PASSED ON APPLE SILICON METAL GPU!            
==========================================================
```

---

## Running the Inference Server

### 1. Launch the Server Daemon
```bash
./build/moecher -m models/qwen3_8_27b_vision_13g/moecher_manifest.json -p 8001 --no-tools --no-think
```

Key CLI Flags:
- `-m, --manifest <path>`: Path to the model manifest JSON file.
- `-p, --port <port>`: Port to bind the HTTP server to (default: `8001`).
- `--no-tools`: Disable system tooling schemas for maximum raw prompt performance.
- `--no-think`: Direct answering mode without extended deliberation blocks.
- `--no-mtp`: Disable speculative drafting (runs pure autoregressive decode).

### 2. Querying the OpenAI-Compatible API
```bash
curl -s http://localhost:8001/v1/chat/completions \
  -H "Content-Type: application/json" \
  -d '{
    "model": "qwen3.8-27B-Vision-13G",
    "messages": [
      {"role": "system", "content": "You are a helpful AI assistant."},
      {"role": "user", "content": "Explain briefly what is Metal Performance Primitives on Apple Silicon."}
    ],
    "max_tokens": 64,
    "temperature": 0.0,
    "stream": false
  }'
```

---

## Performance Summary (Apple M6, 24 GB)

| Engine | Prefill Speed | TTFT (64 prompt) | Decode Speed | Bus Efficiency |
| :--- | :--- | :--- | :--- | :--- |
| **MinnieTheMoEcher (MPP Tiled)** | **143.3 - 172.0 tok/s** | **552.2 ms** | **11.27 - 11.38 tok/s** | **95.1% (146.0 GB/s)** |
| **llama.cpp (llama-bench)** | 143.5 tok/s | 514.8 ms | 10.26 tok/s | 88.7% (136.2 GB/s) |
| **llama.cpp (llama-server)** | 111.2 tok/s | 682.4 ms | 10.14 tok/s | 87.6% (134.6 GB/s) |
| **Apple MLX (mlx_lm.server)** | 53.6 - 87.8 tok/s | 1,431.9 ms | 9.81 tok/s | 94.7% (145.4 GB/s) |

For full benchmarking methodology, microbenchmarks, and architectural details:
- See [BENCHMARKS.md](file:///Users/tinobruno/Developer/MinnieTheMoEcher/BENCHMARKS.md)
- See [WHITEPAPER.md](file:///Users/tinobruno/Developer/MinnieTheMoEcher/WHITEPAPER.md)
