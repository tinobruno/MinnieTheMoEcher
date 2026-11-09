# Raw Engine Output Inspection: `nemotron_3_5_bolt_24gb` — `P2_CODE_FIB`

- **Model**: NVIDIA Nemotron 3.5 Bolt 24GB (74B-A9B IQ2_XXS)
- **Manifest**: `F:\Moecher\models\nemotron_3.5_bolt_24gb\moecher_manifest.json`
- **Test Scenario**: Code Synthesis (Python) (`Software Engineering`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 18:07:12

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `33 tokens` |
| **Time To First Token (TTFT)** | **`153.08 ms`** (`0.123 s`) |
| **Prefill Throughput** | **`269.13 tok/s`** |
| **Autoregressive Decode Speed** | **`46.9 tok/s`** |
| **Generated Tokens** | `320 total` (`0` thinking + `320` content) |
| **Total Wall Time** | `6.998 s` (Decode phase: `6.845 s`) |
| **Finish Reason** | `length` |

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

## 5. Raw Final `content` Output (320 tokens)

```text
Here is a clean, efficient implementation:

```python
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number iteratively.

    The Fibonacci sequence is defined as:
        F(0) = 0, F(1) = 1
        F(n) = F(n-1) + F(n-2) for n > 1

    Args:
        n: The index of the Fibonacci number to calculate. Must be non-negative.

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

**Key features:**
- **Time complexity:** O(n)
- **Space complexity:** O(1)
- **Type hints:** Clear `int` input and output types
- **Docstring:** Documents the algorithm, arguments, return value, and error conditions
- **Edge cases:** Handles `n = 0` and `n = 1` explicitly, and validates negative input

**Example usage:**
```python
print(fibonacci(0))   # 0
print(fibonacci(1))   #
```
