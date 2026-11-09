# Raw Engine Output Inspection: `deepseek_v4_flash_iq2` — `P2_CODE_FIB`

- **Model**: DeepSeek-V4-Flash (314B/11B) — BF16/FP8 Attn + IQ2_XXS
- **Manifest**: `F:\Moecher\models\deepseek_v4_flash_iq2\moecher_manifest.json`
- **Test Scenario**: Code Synthesis (Python) (`Software Engineering`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 16:15:28

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `24 tokens` |
| **Time To First Token (TTFT)** | **`326.69 ms`** (`0.316 s`) |
| **Prefill Throughput** | **`75.86 tok/s`** |
| **Autoregressive Decode Speed** | **`57.57 tok/s`** |
| **Generated Tokens** | `320 total` (`0` thinking + `320` content) |
| **Total Wall Time** | `5.912 s` (Decode phase: `5.585 s`) |
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
Here's a clean Python function that calculates the nth Fibonacci number iteratively:

```python
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number using an iterative approach.

    The Fibonacci sequence is defined as:
    F(0) = 0, F(1) = 1, and F(n) = F(n-1) + F(n-2) for n >= 2.

    Args:
        n (int): The position in the Fibonacci sequence (0-indexed).

    Returns:
        int: The nth Fibonacci number.

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
    
    # Handle base cases
    if n == 0:
        return 0
    if n == 1:
        return 1
    
    # Iterative calculation
    a, b = 0, 1
    for _ in range(2, n + 1):
        a, b = b, a + b
    
    return b
```

This function:

- **Uses type hints** (`n: int` and `-> int`) for clarity
- **Includes a comprehensive docstring** with examples and error documentation
- **Implements an iterative approach** using two variables to track the previous two Fibonacci numbers
- **Handles edge cases**
```
