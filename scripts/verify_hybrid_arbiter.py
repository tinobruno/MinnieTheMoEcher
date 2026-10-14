#!/usr/bin/env python3
"""
Formalized Verification of the Automatic System 1 / System 2 Confidence Arbiter
Replicating the Loan Risk Scenarios from LinkedIn follow-up tests on bare-metal GPU.
"""

import sys
import time
import json
import urllib.request
import urllib.error

BASE_URL = "http://127.0.0.1:8001"

PROMPT_GOOD = """Evaluate applicant for loan risk:
- Applicant: Jane Doe
- Credit Score: 790
- Annual Income: $145,000
- Debt-to-Income (DTI): 18%
- Late Payments (past 24 mo): 0
- Requested Amount: $15,000

Underwriting Decision:"""

PROMPT_BAD = """Evaluate applicant for loan risk:
- Applicant: Jane Doe
- Credit Score: 79
- Annual Income: $14,000
- Debt-to-Income (DTI): 18%
- Late Payments (past 24 mo): 0
- Requested Amount: $150,000

Underwriting Decision:"""

PROMPT_BAD_WITH_RULE = """Evaluate applicant for loan risk:
- Applicant: Jane Doe
- Credit Score: 79
- Annual Income: $14,000
- Debt-to-Income (DTI): 18%
- Late Payments (past 24 mo): 0
- Requested Amount: $150,000

Rule: If credit score is below 600 or requested amount exceeds annual income, the decision must be REJECTED.

Underwriting Decision:"""

CHOICES = ["APPROVED", "REJECTED"]


def test_chat_completion(name: str, prompt: str, zdb_enabled: bool, auto_fallback: bool = True):
    print(f"\n{'='*75}")
    print(f" [TEST] {name}")
    print(f" ZDB Enabled: {zdb_enabled} | Auto Fallback: {auto_fallback}")
    print(f"{'='*75}")

    payload = {
        "model": "minnie-moe",
        "messages": [{"role": "user", "content": prompt}],
        "stream": False,
        "temperature": 0.0,
        "enable_thinking": True,
        "max_tokens": 1024
    }

    if zdb_enabled:
        payload["zero_decode_bypass"] = True
        payload["choices"] = CHOICES
        payload["auto_fallback"] = auto_fallback
        payload["min_confidence"] = 0.70
        payload["max_top1_gap"] = 0.50

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/chat/completions",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error executing request: {e}")
        return None

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    content = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    comp_tokens = usage.get("completion_tokens", 0)
    prefill_ms = usage.get("prefill_sec", 0.0) * 1000.0
    decode_ms = usage.get("decode_sec", 0.0) * 1000.0
    zdb_info = data.get("zero_decode_bypass", {})

    print(f" Wall Latency: {elapsed_ms:.2f} ms")
    print(f" Completion Tokens: {comp_tokens}")
    if zdb_info.get("enabled"):
        arbiter = zdb_info.get("arbiter", {})
        print(f" ZDB Gate Passed: {arbiter.get('gate_passed')}")
        print(f" Path Taken:      {arbiter.get('system_path')}")
        print(f" Top-1 Logit Gap: {arbiter.get('top1_gap', 0.0):.2f}")
        print(f" Confidence:      {zdb_info.get('confidence', 0.0)*100:.2f}%")
        print(f" Instant Choice:  '{zdb_info.get('selected_choice')}'")
    else:
        print(f" System 2 Decode Active (Full Reasoning / Generation)")
        print(f" Response Snippet: {content[:150].strip()}...")

    return {
        "name": name,
        "elapsed_ms": elapsed_ms,
        "comp_tokens": comp_tokens,
        "content": content,
        "zdb_info": zdb_info
    }


def test_decide_endpoint(name: str, prompt: str, auto_fallback: bool = True):
    print(f"\n{'-'*75}")
    print(f" [/v1/decide TEST] {name} (auto_fallback={auto_fallback})")
    print(f"{'-'*75}")

    payload = {
        "prompt": prompt,
        "choices": CHOICES,
        "auto_fallback": auto_fallback,
        "min_confidence": 0.70,
        "max_top1_gap": 0.50,
        "format": "json"
    }

    req_data = json.dumps(payload).encode("utf-8")
    req = urllib.request.Request(
        f"{BASE_URL}/v1/decide",
        data=req_data,
        headers={"Content-Type": "application/json"}
    )

    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            data = json.loads(resp.read().decode("utf-8"))
    except Exception as e:
        print(f"Error executing request: {e}")
        return None

    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    print(f" Decision:        '{data.get('decision')}'")
    print(f" Confidence:      {data.get('confidence', 0.0)*100:.2f}%")
    print(f" Decode Passes:   {data.get('decode_passes')}")
    arb = data.get("arbiter", {})
    print(f" Gate Passed:     {arb.get('gate_passed')}")
    print(f" System Path:     {arb.get('system_path')}")
    print(f" Top-1 Gap:       {arb.get('top1_gap', 0.0):.2f}")
    print(f" Total Latency:   {elapsed_ms:.2f} ms")

    return data


if __name__ == "__main__":
    print("="*75)
    print(" Running Formal Verification for MinnieTheMoEcher Hybrid Arbiter")
    print("="*75)

    # 1. Good applicant (Score 790, Income $145k)
    r1 = test_chat_completion("Good Applicant - Pure ZDB Fast Path", PROMPT_GOOD, zdb_enabled=True, auto_fallback=True)

    # 2. Bad applicant (Score 79, Income $14k, Req $150k) - Hybrid Arbiter enabled
    r2 = test_chat_completion("Bad Applicant - Hybrid Arbiter (Auto Fallback)", PROMPT_BAD, zdb_enabled=True, auto_fallback=True)

    # 3. Bad applicant with explicit rule in context
    r3 = test_chat_completion("Bad Applicant with Explicit Rule - ZDB Fast Path", PROMPT_BAD_WITH_RULE, zdb_enabled=True, auto_fallback=True)

    # 4. Direct /v1/decide endpoint tests
    d1 = test_decide_endpoint("Good Applicant via /v1/decide", PROMPT_GOOD, auto_fallback=True)
    d2 = test_decide_endpoint("Bad Applicant via /v1/decide", PROMPT_BAD, auto_fallback=True)
    d3 = test_decide_endpoint("Bad Applicant with Rule via /v1/decide", PROMPT_BAD_WITH_RULE, auto_fallback=True)

    print("\n" + "="*75)
    print(" Verification Summary Complete")
    print("="*75)
