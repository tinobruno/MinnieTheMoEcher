# Raw Engine Output Inspection: `qwen3_8_27b_vision_13g` — `P1_DIRECT_CHAT`

- **Model**: Qwen 3.8 27B Vision (13G Mixed Quant + MTP)
- **Manifest**: `F:\Moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json`
- **Test Scenario**: Direct Chat (No Thinking) (`Chat & QA`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 16:14:40

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `38 tokens` |
| **Time To First Token (TTFT)** | **`122.55 ms`** (`0.117 s`) |
| **Prefill Throughput** | **`324.37 tok/s`** |
| **Autoregressive Decode Speed** | **`58.99 tok/s`** |
| **Generated Tokens** | `57 total` (`0` thinking + `57` content) |
| **Total Wall Time** | `1.091 s` (Decode phase: `0.968 s`) |
| **Finish Reason** | `stop` |

---

## 2. Automated Quality & Hallucination Audit: **PASS (3/3 checks passed)**

- **PASS** — `Anti-Loop / Repetition Check`: Zero n-gram repetition loops detected
- **PASS** — `UTF-8 Token Integrity`: Clean UTF-8 token decoding
- **PASS** — `Concept Accuracy (Mutex vs Semaphore)`: Correctly contrasts ownership/locking vs counting/signaling

---

## 3. Exact Prompt Sent to Engine

- **Parameters**: `temperature=0.0`, `max_tokens=120`, `enable_thinking=False`, `thinking_budget=0`

```text
Explain the difference between a mutex and a semaphore in two concise sentences.
```

---

## 4. Raw `<think>` / `reasoning_content` Trace (0 tokens)

```text
(No thinking trace emitted / thinking disabled)
```

---

## 5. Raw Final `content` Output (57 tokens)

```text
A mutex is a binary lock that ensures only one thread can access a resource at a time, typically used to protect shared data. A semaphore is a counter-based mechanism that allows a specific number of threads to access a resource simultaneously, making it useful for managing access to pools of resources.
```
