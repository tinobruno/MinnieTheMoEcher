# Raw Engine Output Inspection: `qwen3_8_27b_q4` — `P2_CODE_FIB`

- **Model**: Qwen 3.8 27B Dense — Q4_K / BF16
- **Manifest**: `F:\Moecher\models\qwen3_8_27b_q4\moecher_manifest.json`
- **Test Scenario**: Code Synthesis (Python) (`Software Engineering`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 16:13:44

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `43 tokens` |
| **Time To First Token (TTFT)** | **`132.94 ms`** (`0.127 s`) |
| **Prefill Throughput** | **`339.57 tok/s`** |
| **Autoregressive Decode Speed** | **`84.37 tok/s`** |
| **Generated Tokens** | `209 total` (`0` thinking + `209` content) |
| **Total Wall Time** | `2.612 s` (Decode phase: `2.479 s`) |
| **Finish Reason** | `stop` |

---

## 2. Automated Quality & Hallucination Audit: **PASS (3/3 checks passed)**

- **PASS** — `Anti-Loop / Repetition Check`: Zero n-gram repetition loops detected
- **PASS** — `UTF-8 Token Integrity`: Clean UTF-8 token decoding
- **PASS** — `Live Python Execution & Assertion Test`: Executed `fibonacci(10)==55` and `fibonacci(20)==6765` -> PASS

---

## 3. Exact Prompt Sent to Engine

- **Parameters**: `temperature=0.0`, `max_tokens=320`, `enable_thinking=False`, `thinking_budget=0`

```text
Write a clean Python function to calculate the nth Fibonacci number iteratively with type hints and docstring.
```

---

## 4. Raw `<think>` / `reasoning_content` Trace (0 tokens)

```text
(No thinking trace emitted / thinking disabled)
```

---

## 5. Raw Final `content` Output (209 tokens)

```text
```python
def fibonacci(n: int) -> int:
    """Calculate the nth Fibonacci number iteratively.

    The Fibonacci sequence is defined as:
        F(0) = 0
        F(1) = 1
        F(n) = F(n-1) + F(n-2) for n >= 2

    Args:
        n: A non-negative integer representing the position in the Fibonacci sequence.

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
    
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
```
```
