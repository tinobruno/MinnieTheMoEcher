# Raw Engine Output Inspection: `qwen3_8_27b_vision_13g` — `P3_HTML_CALC`

- **Model**: Qwen 3.8 27B Vision (13G Mixed Quant + MTP)
- **Manifest**: `F:\Moecher\models\qwen3_8_27b_vision_13g\moecher_manifest.json`
- **Test Scenario**: Interactive Web UI Code (`Frontend Architecture`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 16:14:40

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `37 tokens` |
| **Time To First Token (TTFT)** | **`133.34 ms`** (`0.103 s`) |
| **Prefill Throughput** | **`360.64 tok/s`** |
| **Autoregressive Decode Speed** | **`73.56 tok/s`** |
| **Generated Tokens** | `1229 total` (`28` thinking + `1200` content) |
| **Total Wall Time** | `16.872 s` (Decode phase: `16.739 s`) |
| **Finish Reason** | `length` |

---

## 2. Automated Quality & Hallucination Audit: **PASS (3/3 checks passed)**

- **PASS** — `Anti-Loop / Repetition Check`: Zero n-gram repetition loops detected
- **PASS** — `UTF-8 Token Integrity`: Clean UTF-8 token decoding
- **PASS** — `HTML/CSS/JS Structure Check`: Valid HTML calculator markup generated

---

## 3. Exact Prompt Sent to Engine

- **Parameters**: `temperature=0.0`, `max_tokens=1200`, `enable_thinking=True`, `thinking_budget=200`

```text
Write a clean, responsive single-file HTML calculator with modern styling and interactive JavaScript.
```

---

## 4. Raw `<think>` / `reasoning_content` Trace (28 tokens)

```text
Create a single-file HTML calculator with modern styling, responsive design, and interactive JS. Provide the complete code along with a brief explanation.

```

---

## 5. Raw Final `content` Output (1200 tokens)

```text


# Modern Single-File HTML Calculator

Here's a complete, self-contained calculator with a sleek dark-mode design, keyboard support, and a fully responsive layout:

```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Calculator</title>
  <style>
    * {
      margin: 0;
      padding: 0;
      box-sizing: border-box;
    }

    :root {
      --bg: #1e1e2e;
      --calc-bg: #181825;
      --display-bg: #11111b;
      --btn: #313244;
      --btn-hover: #45475a;
      --btn-op: #58775e;
      --btn-eq: #89b4fa;
      --text: #cdd6f4;
      --text-dim: #7f849c;
      --radius: 16px;
    }

    body {
      font-family: "Segoe UI", system-ui, -apple-system, sans-serif;
      background: var(--bg);
      min-height: 100vh;
      display: flex;
      align-items: center;
      justify-content: center;
      padding: 16px;
    }

    .calculator {
      width: 100%;
      max-width: 340px;
      background: var(--calc-bg);
      border-radius: 24px;
      padding: 20px;
      box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5);
    }

    /* ---------- Display ---------- */
    .display {
      background: var(--display-bg);
      border-radius: var(--radius);
      padding: 16px 18px;
      margin-bottom: 18px;
      text-align: right;
      overflow: hidden;
    }

    .display .expression {
      font-size: 1rem;
      color: var(--text-dim);
      min-height: 1.2em;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
    }

    .display .result {
      font-size: 2.2rem;
      font-weight: 600;
      color: var(--text);
      margin-top: 4px;
      white-space: nowrap;
      overflow: hidden;
      text-overflow: ellipsis;
      transition: font-size 0.15s ease;
    }

    .display .result.shrink {
      font-size: 1.4rem;
    }

    /* ---------- Buttons ---------- */
    .buttons {
      display: grid;
      grid-template-columns: repeat(4, 1fr);
      gap: 10px;
    }

    button {
      border: none;
      border-radius: var(--radius);
      padding: 18px 0;
      font-size: 1.15rem;
      font-weight: 600;
      color: var(--text);
      background: var(--btn);
      cursor: pointer;
      transition: transform 0.1s ease, background 0.15s ease;
      -webkit-tap-highlight-color: transparent;
    }

    button:hover {
      background: var(--btn-hover);
    }

    button:active {
      transform: scale(0.94);
    }

    button.op {
      background: var(--btn-op);
      color: #fff;
    }

    button.eq {
      background: var(--btn-eq);
      color: #11111b;
    }

    button.fn {
      background: #45475a;
      color: var(--text-dim);
    }

    /* ---------- Responsive tweaks ---------- */
    @media (max-width: 380px) {
      .calculator {
        padding: 14px;
        border-radius: 18px;
      }
      .buttons {
        gap: 8px;
      }
      button {
        padding: 14px 0;
        font-size: 1rem;
      }
      .display .result {
        font-size: 1.8rem;
      }
    }
  </style>
</head>
<body>

  <div class="calculator">
    <div class="display">
      <div class="expression" id="expression"></div>
      <div class="result" id="result">0</div>
    </div>

    <div class="buttons">
      <button class="fn" data-action="clear">AC</button>
      <button class="fn" data-action="delete">⌫</button>
      <button class="op" data-op="÷">÷</button>
      <button class="op" data-op="×">×</button>

      <button data-num="
```
