# Raw Engine Output Inspection: `deepseek_v41_mixed` — `P2_CODE_FIB`

- **Model**: DeepSeek-V4.1-Flash (552B/14B) — Mixed NVFP4 + IQ2
- **Manifest**: `F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json`
- **Test Scenario**: Code Synthesis (Python) (`Software Engineering`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 15:41:13

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `31 tokens` |
| **Time To First Token (TTFT)** | **`1284.51 ms`** (`1.278 s`) |
| **Prefill Throughput** | **`24.26 tok/s`** |
| **Autoregressive Decode Speed** | **`12.59 tok/s`** |
| **Generated Tokens** | `220 total` (`0` thinking + `220` content) |
| **Total Wall Time** | `18.757 s` (Decode phase: `17.472 s`) |
| **Finish Reason** | `length` |

---

## 2. Automated Quality & Hallucination Audit: **PASS (3/3 checks passed)**

- **PASS** — `Anti-Loop / Repetition Check`: Zero n-gram repetition loops detected
- **PASS** — `UTF-8 Token Integrity`: Clean UTF-8 token decoding
- **PASS** — `Live Python Execution & Assertion Test`: Executed `fibonacci(10)==55` and `fibonacci(20)==6765` -> PASS

---

## 3. Exact Prompt Sent to Engine

- **Parameters**: `temperature=0.0`, `max_tokens=220`, `enable_thinking=False`, `thinking_budget=0`

```text
Write a clean Python function to calculate the nth Fibonacci number iteratively with type hints and docstring.
```

---

## 4. Raw `<think>` / `reasoning_content` Trace (0 tokens)

```text
(No thinking trace emitted / thinking disabled)
```

---

## 5. Raw Final `content` Output (220 tokens)

```text
Here's a clean Python function to calculate the nth Fibonacci number iteratively:

```python
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number iteratively.

    The Fibonacci sequence is defined as:
        F(0) = 0
        F(1) = 1
        F(n) = F(n-1) + F(n-2) for n >= 2

    Args:
        n: The index in the Fibonacci sequence (must be >= 0).

    Returns:
        The nth Fibonacci number.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    if n == 0:
        return 0
    if n == 1:
        return 1

    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr
```

**Key points:**

- **Type hints**:
```
