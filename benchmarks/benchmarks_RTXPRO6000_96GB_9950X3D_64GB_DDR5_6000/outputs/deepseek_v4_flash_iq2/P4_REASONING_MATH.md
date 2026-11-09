# Raw Engine Output Inspection: `deepseek_v4_flash_iq2` — `P4_REASONING_MATH`

- **Model**: DeepSeek-V4-Flash (314B/11B) — BF16/FP8 Attn + IQ2_XXS
- **Manifest**: `F:\Moecher\models\deepseek_v4_flash_iq2\moecher_manifest.json`
- **Test Scenario**: Step-by-Step Reasoning (`Deep Reasoning`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 15:59:57

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `42 tokens` |
| **Time To First Token (TTFT)** | **`443.63 ms`** (`0.434 s`) |
| **Prefill Throughput** | **`96.84 tok/s`** |
| **Autoregressive Decode Speed** | **`65.27 tok/s`** |
| **Generated Tokens** | `311 total` (`113` thinking + `197` content) |
| **Total Wall Time** | `5.213 s` (Decode phase: `4.769 s`) |
| **Finish Reason** | `stop` |

---

## 2. Automated Quality & Hallucination Audit: **PASS (3/3 checks passed)**

- **PASS** — `Anti-Loop / Repetition Check`: Zero n-gram repetition loops detected
- **PASS** — `UTF-8 Token Integrity`: Clean UTF-8 token decoding
- **PASS** — `Mathematical Ground Truth ($0.05 / 5 cents)`: Verified exact algebraic solution ($0.05 ball + $1.05 bat = $1.10)

---

## 3. Exact Prompt Sent to Engine

- **Parameters**: `temperature=0.0`, `max_tokens=450`, `enable_thinking=True`, `thinking_budget=500`

```text
A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. How much does the ball cost? Show step-by-step logic.
```

---

## 4. Raw `<think>` / `reasoning_content` Trace (113 tokens)

```text
We need answer classic riddle. Need show step-by-step logic. Need be careful: bat + ball = 1.10, bat = ball + 1.00. Solve: ball = x, bat = x+1.00, total 2x+1 = 1.10 => 2x=0.10 => x=0.05. Ball costs 5 cents, bat costs $1.05. Need maybe mention common wrong answer 10 cents. User asks "Show step-by-step logic." We'll provide concise.
```

---

## 5. Raw Final `content` Output (197 tokens)

```text
Let the ball cost be **x** dollars.

The bat costs **$1.00 more than the ball**, so the bat costs:

**x + 1.00**

Together, the bat and ball cost **$1.10**:

**x + (x + 1.00) = 1.10**

Combine like terms:

**2x + 1.00 = 1.10**

Subtract 1.00 from both sides:

**2x = 0.10**

Divide both sides by 2:

**x = 0.05**

So the ball costs **$0.05**, or **5 cents**.

Then the bat costs:

**0.05 + 1.00 = 1.05**

Check: **$1.05 + $0.05 = $1.10**, and the bat costs $1.00 more than the ball. ✅

**Answer: The ball costs 5 cents.**
```
