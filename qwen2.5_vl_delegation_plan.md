# Implementation Plan: In-Process Qwen2.5-VL-3B Vision Delegation in Moecher

## Overview
This plan establishes a compound AI architecture inside `moecher`:
1. **Primary Brain**: **DeepSeek-V4-Flash** (MoE, 100% VRAM-resident at ~87.3 GB) handles general conversation, reasoning, code generation, tool calls, and Three.js 3D reconstruction.
2. **Dedicated Eyes**: **Qwen2.5-VL-3B-Instruct** (quantized to ~2.2 GB) handles native, grounded multimodal perception (vehicle make/model recognition, OCR license plates, scene structural breakdown).
3. **Transparent C++ Orchestration**: Pure in-process C++ execution in `src/server_single.cpp` with zero Python runtime dependencies. All weights and manifests are self-contained in `models/frankenstin/`.

---

## Progress Dashboard

| Phase | Description | Status |
| :--- | :--- | :--- |
| **Phase 1** | Model Acquisition & Binary Quantization (`models/frankenstin/vision/`) | `COMPLETED` |
| **Phase 2** | Manifest & Binary Layout Specification | `COMPLETED` |
| **Phase 3** | C++ Dual-Engine Integration in `src/server_single.cpp` | `IN_PROGRESS` |
| **Phase 4** | Transparent In-Process Perception Delegation | `PENDING` |
| **Phase 5** | Compilation & Verification on Mercedes, BMW & Control Images | `PENDING` |

---

## Detailed Step-by-Step Task List

### Phase 1: Model Acquisition & Binary Quantization
- [x] **Step 1.1**: Download `Qwen/Qwen2.5-VL-3B-Instruct` snapshot from Hugging Face.
- [x] **Step 1.2**: Write `scripts/convert_qwen2.5_vl_3b.py` to convert and quantize Qwen2.5-VL-3B:
  - Language model backbone to INT4 block-32 dense format (`models/frankenstin/vision/attention_dense_layers.bin`, ~3.28 GB).
  - ViT Vision Tower + merger weights in BF16 included in `attention_dense_layers.bin`.
  - Tokenizer and processor configuration files copied to `models/frankenstin/vision/`.
- [x] **Step 1.3**: Validate output tensor shapes, norms, and memory footprint (~3.28 GB).

### Phase 2: Manifest & Binary Layout Specification
- [x] **Step 2.1**: Generate `models/frankenstin/vision/moecher_manifest.json` describing the Qwen2.5-VL-3B delegate (968 tensors).
- [x] **Step 2.2**: Update `models/frankenstin/moecher_manifest.json` to link the delegate:
  ```json
  "vision_delegate": {
    "manifest": "vision/moecher_manifest.json"
  }
  ```
- [x] **Step 2.3**: Verify total combined VRAM footprint calculation (~90.6 GB / 96 GB: DeepSeek 87.3 GB + Qwen 3.3 GB).

### Phase 3: C++ Dual-Engine Integration in `src/server_single.cpp`
- [ ] **Step 3.1**: Add `std::unique_ptr<MoecherEngine> vision_engine_;` to `MoecherServer` / global engine state.
- [ ] **Step 3.2**: On server startup, if `manifest.contains("vision_delegate")`, load `vision_engine_` using the vision delegate manifest in a dedicated CUDA stream.
- [ ] **Step 3.3**: Ensure clean memory cohabitation and verify zero CUDA allocation conflicts between DeepSeek and Qwen.

### Phase 4: Transparent In-Process Perception Delegation
- [ ] **Step 4.1**: In `/v1/chat/completions` (both SSE streaming and non-streaming), detect incoming image attachments.
- [ ] **Step 4.2**: Extract the image and execute a targeted visual analysis pass on `vision_engine_`:
  - Prompt: `"Describe this image in detail. Identify any vehicles (make, model, color, year), read exact text and license plates, and detail structural parts."`
- [ ] **Step 4.3**: Transparently inject the resulting `<visual_perception>` block into DeepSeek's context before the user's prompt.
- [ ] **Step 4.4**: DeepSeek streams the final response directly to the client (or generates Three.js 3D models).
- [ ] **Step 4.5**: Ensure multi-turn continuity: For subsequent user messages in the same conversation, preserve the `<visual_perception>` in the message history so follow-ups run with zero visual overhead and zero hallucination.

### Phase 5: Compilation & Verification
- [ ] **Step 5.1**: Compile `moecher` via `make -C build -j$(nproc)`.
- [ ] **Step 5.2**: Launch `./build/moecher --manifest models/frankenstin/moecher_manifest.json --port 8001`.
- [ ] **Step 5.3**: Run live verification tests:
  - Test A: User's Mercedes image -> Must recognize as Mercedes-Benz, not Ferrari.
  - Test B: Blue BMW image -> Must recognize BMW 3/5 Series, blue color, and OCR plate `B 58 BPS`.
  - Test C: Control blank/white image -> Must state that no vehicle or empty image is shown.
  - Test D: Multi-turn follow-up -> Confirm immediate, coherent answers without hallucination.

---

## Checkpoint & Recovery State
* **Current State**: Phase 1 & 2 completed. Binary quantized (~3.28 GB) and manifest wired in `models/frankenstin/moecher_manifest.json`.
* **Resume Point**: Step 3.1 (C++ Dual-Engine Integration in `src/server_single.cpp`).
