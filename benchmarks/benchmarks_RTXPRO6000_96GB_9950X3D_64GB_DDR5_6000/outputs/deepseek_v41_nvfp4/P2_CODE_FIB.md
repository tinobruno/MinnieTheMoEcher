# Raw Engine Output Inspection: `deepseek_v41_nvfp4` — `P2_CODE_FIB`

- **Model**: DeepSeek-V4.1-Flash (552B/14B) — Pure NVFP4
- **Manifest**: `F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest.json`
- **Test Scenario**: Code Synthesis (Python) (`Software Engineering`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 17:06:10

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `31 tokens` |
| **Time To First Token (TTFT)** | **`8953.72 ms`** (`8.949 s`) |
| **Prefill Throughput** | **`3.46 tok/s`** |
| **Autoregressive Decode Speed** | **`4.07 tok/s`** |
| **Generated Tokens** | `196 total` (`0` thinking + `196` content) |
| **Total Wall Time** | `57.136 s` (Decode phase: `48.187 s`) |
| **Finish Reason** | `stop` |

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

## 5. Raw Final `content` Output (196 tokens)

```text
```python
def fibonacci(n: int) -> int:
    """Return the nth Fibonacci number using an iterative approach.

    The sequence is defined as:
        F(0) = 0, F(1) = 1, F(n) = F(n-1) + F(n-2) for n > 1.

    Args:
        n: The index of the Fibonacci number to compute. Must be >= 0.

    Returns:
        The nth Fibonacci number.

    Raises:
        ValueError: If n is negative.

    Examples:
        >>> fibonacci(0)
        0
        >>> fibonacci(1)
        1
        >>> fibonacci(10)
        55
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")

    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
```
```
