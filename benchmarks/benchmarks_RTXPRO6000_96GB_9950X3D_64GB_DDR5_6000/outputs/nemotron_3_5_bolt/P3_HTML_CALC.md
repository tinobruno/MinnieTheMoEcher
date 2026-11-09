# Raw Engine Output Inspection: `nemotron_3_5_bolt` — `P3_HTML_CALC`

- **Model**: NVIDIA Nemotron 3.5 Bolt (74B-A9B NVFP4)
- **Manifest**: `F:\Moecher\models\nemotron_3.5_bolt\moecher_manifest_24gb.json`
- **Test Scenario**: Interactive Web UI Code (`Frontend Architecture`)
- **Hardware**: NVIDIA RTX PRO 6000 Blackwell Max-Q Workstation Edition (96 GB VRAM) + AMD Ryzen 9 9950X3D 16-Core Processor + 64 GB DDR5-6000
- **Timestamp**: 2026-11-01 17:04:38

---

## 1. Performance Telemetry

| Metric | Measured Value |
| :--- | :--- |
| **Prompt Tokens Evaluated** | `35 tokens` |
| **Time To First Token (TTFT)** | **`87.8 ms`** (`0.081 s`) |
| **Prefill Throughput** | **`433.33 tok/s`** |
| **Autoregressive Decode Speed** | **`54.59 tok/s`** |
| **Generated Tokens** | `1280 total` (`79` thinking + `1200` content) |
| **Total Wall Time** | `23.554 s` (Decode phase: `23.466 s`) |
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

## 4. Raw `<think>` / `reasoning_content` Trace (79 tokens)

```text
The user wants a clean, responsive single-file HTML calculator with modern styling and interactive JavaScript. I'll create a complete HTML file with embedded CSS and JavaScript.

Key features:
- Clean, modern styling
- Responsive design
- Basic calculator operations (+, -, *, /, =, C, AC, decimal)
- Interactive JavaScript
- Single file

Let me build this.
```

---

## 5. Raw Final `content` Output (1200 tokens)

```text
```html
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Modern Calculator</title>
    <style>
        * {
            margin: 0;
            padding: 0;
            box-sizing: border-box;
        }

        body {
            font-family: 'Segoe UI', system-ui, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            padding: 20px;
        }

        .calculator {
            background: #ffffff;
            border-radius: 24px;
            box-shadow: 0 20px 40px rgba(0, 0, 0, 0.2);
            width: 100%;
            max-width: 400px;
            overflow: hidden;
        }

        .display {
            background: #f8f9fa;
            padding: 24px;
            text-align: right;
            border-bottom: 1px solid #e9ecef;
        }

        .previous-operand {
            font-size: 14px;
            color: #6c757d;
            min-height: 20px;
            word-wrap: break-word;
        }

        .current-operand {
            font-size: 32px;
            font-weight: 300;
            color: #212529;
            min-height: 40px;
            word-wrap: break-word;
        }

        .buttons {
            display: grid;
            grid-template-columns: repeat(4, 1fr);
            gap: 10px;
            padding: 20px;
        }

        button {
            padding: 20px 0;
            font-size: 20px;
            font-weight: 500;
            border: none;
            border-radius: 12px;
            cursor: pointer;
            transition: all 0.15s ease;
            background: #ffffff;
            color: #212529;
        }

        button:hover {
            background: #f8f9fa;
            transform: translateY(-1px);
        }

        button:active {
            transform: translateY(0);
            background: #e9ecef;
        }

        .operator {
            color: #667eea;
            font-weight: 600;
        }

        .operator:hover {
            background: #f0f2ff;
        }

        .equals {
            background: #667eea;
            color: #ffffff;
            grid-column: span 2;
        }

        .equals:hover {
            background: #5568d3;
        }

        .clear {
            color: #dc3545;
        }

        .clear:hover {
            background: #f8d7da;
        }

        .all-clear {
            color: #dc3545;
        }

        .all-clear:hover {
            background: #f8d7da;
        }

        .decimal {
            color: #212529;
        }

        @media (max-width: 480px) {
            .calculator {
                border-radius: 16px;
            }

            .display {
                padding: 16px;
            }

            .current-operand {
                font-size: 28px;
            }

            .buttons {
                gap: 8px;
                padding: 16px;
            }

            button {
                padding: 16px 0;
                font-size: 18px;
                border-radius: 10px;
            }
        }

        @media (max-width: 360px) {
            .current-operand {
                font-size: 24px;
            }

            button {
                padding: 14px 0;
                font-size: 16px;
            }
        }
    </style>
</head>
<body>
    <div class="calculator">
        <div class="display">
            <div class="previous-operand" id="previousOperand"></div>
            <div class="current-operand" id="currentOperand">0</div>
        </div>
        <div class="buttons">
            <button class="all-clear" data-action="clear">AC</button>
            <button class="clear" data-action="delete">DEL</button>
            <button class="operator" data-action="percentage">%</button>
            <button class="operator" data-operation="/">÷</button>
            
            <button data-number="7">7</button>
            <button data-number="8">8</button>
            <button data-number="9">9</button>
            <button class="operator" data-operation="*">×</button>
            
            <button data-number="4">4</button>
            <button data-number="5">5</button>
            <button data-number="6">6</button>
            <button class="operator" data-operation="-">−</button>
            
            <button data-number="1">1</button>
            <button data-number="2">2</button>
            <button data-number="3">3</button>
            <button class="operator" data-operation="+">+</button>
            
```
