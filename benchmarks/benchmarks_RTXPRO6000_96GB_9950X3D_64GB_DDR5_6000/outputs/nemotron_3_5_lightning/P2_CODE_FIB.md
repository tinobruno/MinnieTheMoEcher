# Raw Engine Output Inspection: `nemotron_3_5_lightning` — `P2_CODE_FIB`

- **Model**: NVIDIA Nemotron 3.5 Lightning Vision (30B-A3B NVFP4)
- **Manifest**: `F:\Moecher\models\nemotron_3.5_lightning\moecher_manifest.json`
- **Test Scenario**: Code Synthesis (Python) (`Software Engineering`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 17:03:18

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `33 tokens` |
| **Time To First Token (TTFT)** | **`60.6 ms`** (`0.055 s`) |
| **Prefill Throughput** | **`595.2 tok/s`** |
| **Autoregressive Decode Speed** | **`112.12 tok/s`** |
| **Generated Tokens** | `320 total` (`0` thinking + `320` content) |
| **Total Wall Time** | `2.924 s` (Decode phase: `2.863 s`) |
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
Here is a clean Python function to calculate the nth Fibonacci number iteratively.

```python
def fibonacci(n: int) -> int:
    """
    Calculate the nth Fibonacci number iteratively.

    The Fibonacci sequence is defined by the recurrence relation:
    F(n) = F(n-1) + F(n-2), with seed values F(0) = 0 and F(1) = 1.

    Args:
        n (int): The position in the Fibonacci sequence (must be non-negative).

    Returns:
        int: The nth Fibonacci number.

    Raises:
        ValueError: If n is negative.
    """
    if n < 0:
        raise ValueError("Input must be a non-negative integer.")

    # Base cases
    if n == 0:
        return 0
    if n == 1:
        return 1

    # Iterative calculation
    prev, curr = 0, 1
    for _ in range(2, n + 1):
        prev, curr = curr, prev + curr

    return curr
```

### Key Features:
*   **Type Hints:** Uses `int` for both the input argument and return type.
*   **Docstring:** Includes a description, the mathematical definition, argument explanation, return value explanation, and exception handling.
*   **Iterative Approach:** Uses a simple `for` loop with tuple unpacking to update variables, which is memory efficient (O(1) space complexity) compared to recursion
```
