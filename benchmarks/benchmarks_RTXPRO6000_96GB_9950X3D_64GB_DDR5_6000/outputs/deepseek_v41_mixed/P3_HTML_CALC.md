# Raw Engine Output Inspection: `deepseek_v41_mixed` — `P3_HTML_CALC`

- **Model**: DeepSeek-V4.1-Flash (552B/14B) — Mixed NVFP4 + IQ2
- **Manifest**: `F:\Moecher\models\deepseek_v41_nvfp4\moecher_manifest_mixed.json`
- **Test Scenario**: Interactive Web UI Code (`Frontend Architecture`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 15:41:13

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `52 tokens` |
| **Time To First Token (TTFT)** | **`1697.63 ms`** (`1.683 s`) |
| **Prefill Throughput** | **`30.9 tok/s`** |
| **Autoregressive Decode Speed** | **`16.65 tok/s`** |
| **Generated Tokens** | `1401 total` (`200` thinking + `1200` content) |
| **Total Wall Time** | `85.866 s` (Decode phase: `84.169 s`) |
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

## 4. Raw `<think>` / `reasoning_content` Trace (200 tokens)

```text
The user wants a single-file HTML calculator with modern styling and interactive JavaScript. I need to write clean, responsive code. Let me build a nice calculator UI with display, buttons, keyboard support, and proper calculation logic.

Let me design:
- Modern dark/light theme, glassmorphism or clean card design
- Responsive grid layout for buttons
- Display showing current input and result
- Support operators + - × ÷, decimal, %, clear, backspace, equals
- Keyboard support
- Handle edge cases like division by zero, multiple decimals, etc.

Let me write the code carefully.

Structure:
```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Calculator</title>
<style>...</style>
</head>
<body>
<div class="calculator">
  <div class="display">
    <div class="expression"></div>

```

---

## 5. Raw Final `content` Output (1200 tokens)

```text
Here's a clean, responsive single-file calculator with modern styling and interactive JavaScript:

```html
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Modern Calculator</title>
<style>
  :root {
    --bg: #0f172a;
    --card: #1e293b;
    --display: #0b1220;
    --btn: #334155;
    --btn-hover: #475569;
    --accent: #38bdf8;
    --accent-hover: #0ea5e9;
    --text: #f1f5f9;
    --muted: #94a3b8;
    --danger: #ef4444;
    --radius: 16px;
  }

  * {
    box-sizing: border-box;
    margin: 0;
    padding: 0;
    font-family: 'Segoe UI', system-ui, -apple-system, sans-serif;
  }

  body {
    min-height: 100vh;
    display: flex;
    align-items: center;
    justify-content: center;
    background: radial-gradient(circle at top left, #1e293b, #0f172a 60%);
    padding: 16px;
  }

  .calculator {
    width: 100%;
    max-width: 360px;
    background: var(--card);
    border-radius: var(--radius);
    padding: 20px;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255,255,255,0.05);
    border: 1px solid rgba(255,255,255,0.08);
  }

  .display {
    background: var(--display);
    border-radius: 12px;
    padding: 16px;
    margin-bottom: 20px;
    text-align: right;
    min-height: 90px;
    display: flex;
    flex-direction: column;
    justify-content: flex-end;
    overflow: hidden;
    border: 1px solid rgba(255,255,255,0.05);
  }

  .expression {
    font-size: 0.9rem;
    color: var(--muted);
    min-height: 20px;
    word-break: break-all;
    line-height: 1.4;
  }

  .result {
    font-size: 2.2rem;
    font-weight: 600;
    color: var(--text);
    word-break: break-all;
    line-height: 1.2;
    margin-top: 4px;
  }

  .keys {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 10px;
  }

  button {
    border: none;
    background: var(--btn);
    color: var(--text);
    font-size: 1.1rem;
    font-weight: 500;
    padding: 16px 8px;
    border-radius: 10px;
    cursor: pointer;
    transition: background 0.15s ease, transform 0.1s ease;
    user-select: none;
    -webkit-tap-highlight-color: transparent;
  }

  button:hover {
    background: var(--btn-hover);
  }

  button:active {
    transform: scale(0.95);
  }

  button.accent {
    background: var(--accent);
    color: #0f172a;
    font-weight: 600;
  }

  button.accent:hover {
    background: var(--accent-hover);
  }

  button.danger {
    background: var(--danger);
    color: #fff;
  }

  button.danger:hover {
    background: #dc2626;
  }

  button.span-2 {
    grid-column: span 2;
  }

  @media (max-width: 400px) {
    .calculator { padding: 14px; }
    button { padding: 14px 6px; font-size: 1rem; }
    .result { font-size: 1.8rem; }
  }
</style>
</head>
<body>

<div class="calculator">
  <div class="display">
    <div class="expression" id="expression"></div>
    <div class="result" id="result">0</div>
  </div>

  <div class="keys">
    <button class="danger" data-action="clear">AC</button>
    <button data-action="backspace">⌫</button>
    <button data-action="percent">%</button>
    <button class="accent" data-action="divide">÷</button>

    <button data-value="7">7</button>
    <button data-value="8">8</button>
    <button data-value="9">9</button>
    <button class="accent" data-action="multiply">×</button>

    <button data-value="4">4</button>
    <button data-value="5">5</button>
    <button data-value="6">6</button>
    <button class="accent" data-action="subtract">−</button>

    <button data-value="1">1</button>
    <button data-value="2">2</button>
    <button data-value="3">3</button>
    <button class="accent" data-action="add">+</button>

    <button
```
