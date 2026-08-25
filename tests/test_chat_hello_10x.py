#!/usr/bin/env python3
"""
Runs the chat.py 'Hello' test 10 consecutive times with:
- temp: 0.6
- budget: 10000
- max_tokens: 10000
and checks if the think end token is sent on each run.
"""

import sys
import os
import time

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat import chat_completion_stream

def run_single_test(iter_num: int):
    url = "http://localhost:8001"
    temperature = 0.6
    thinking_budget = 10000
    max_tokens = 10000
    prompt_text = "Hello"

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt_text}
    ]

    print(f"\n{'='*75}")
    print(f"▶️ RUN {iter_num}/10 — Prompt: '{prompt_text}' (temp={temperature}, budget={thinking_budget})")
    print(f"{'='*75}")

    reasoning_chunks = []
    content_chunks = []
    has_reasoning = False
    has_content = False
    think_end_detected = False

    t_start = time.perf_counter()
    ttft = None

    stream = chat_completion_stream(
        url=url,
        messages=messages,
        max_tokens=max_tokens,
        temperature=temperature,
        thinking="enabled",
        reasoning_effort="high",
        thinking_budget=thinking_budget
    )

    for chunk in stream:
        if ttft is None:
            ttft = time.perf_counter() - t_start

        if isinstance(chunk, dict):
            if chunk.get("type") == "reasoning":
                has_reasoning = True
                reasoning_chunks.append(chunk["content"])
                print(chunk["content"], end="", flush=True)
            elif "usage" in chunk:
                pass
        elif isinstance(chunk, str):
            if not has_content and has_reasoning:
                think_end_detected = True
                print("\n\n" + "-"*30 + " [THINK END -> CONTENT] " + "-"*30 + "\n")
            has_content = True
            content_chunks.append(chunk)
            print(chunk, end="", flush=True)

    t_end = time.perf_counter()
    duration = t_end - t_start

    full_reasoning = "".join(reasoning_chunks).strip()
    full_content = "".join(content_chunks).strip()

    passed = (think_end_detected or (has_content and not has_reasoning)) and bool(full_content)

    print(f"\n\n{'─'*75}")
    print(f"Run {iter_num}: TTFT={ttft:.3f}s | Duration={duration:.2f}s | Think Chunks={len(reasoning_chunks)} | Content Chunks={len(content_chunks)} | Think End={'✅ YES' if passed else '❌ NO'}")
    print(f"{'─'*75}")

    return {
        "run": iter_num,
        "ttft": ttft,
        "duration": duration,
        "think_chunks": len(reasoning_chunks),
        "content_chunks": len(content_chunks),
        "think_end_detected": passed,
        "reasoning": full_reasoning,
        "content": full_content
    }

def main():
    print(f"🚀 Running 10 consecutive tests with chat.py ...")
    results = []

    for i in range(1, 11):
        res = run_single_test(i)
        results.append(res)
        time.sleep(0.5)

    print("\n" + "=" * 80)
    print("📋 SUMMARY TABLE: 10 CONSECUTIVE RUNS")
    print("=" * 80)
    print(f"{'Run':<5} | {'TTFT (s)':<10} | {'Duration (s)':<14} | {'Think Chunks':<14} | {'Content Chunks':<16} | {'Think End Sent':<15}")
    print("-" * 80)

    all_passed = True
    for r in results:
        status = "✅ YES" if r["think_end_detected"] else "❌ NO"
        if not r["think_end_detected"]:
            all_passed = False
        print(f"{r['run']:<5} | {r['ttft']:<10.4f} | {r['duration']:<14.2f} | {r['think_chunks']:<14} | {r['content_chunks']:<16} | {status:<15}")

    print("=" * 80)
    if all_passed:
        print("🎉 ALL 10/10 RUNS SUCCESSFULLY DETECTED AND PROCESSED THE THINK END TOKEN!")
        return 0
    else:
        print("⚠️ SOME RUNS FAILED TO DETECT THINK END TOKEN.")
        return 1

if __name__ == "__main__":
    sys.exit(main())
