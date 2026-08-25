#!/usr/bin/env python3
"""
Test script for verifying 'Hello' prompt with:
- temperature: 0.6
- thinking_budget: 10000
- max_tokens: 10000
using chat.py's chat_completion_stream, verifying that the think block finishes
and transitions to content (think end).
"""

import sys
import os
import time

# Add root directory to sys.path to import chat.py
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat import chat_completion_stream

def test_hello_think_end():
    url = "http://localhost:8001"
    temperature = 0.6
    thinking_budget = 10000
    max_tokens = 10000
    prompt_text = "Hello"

    messages = [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": prompt_text}
    ]

    print("=" * 70)
    print(f"🚀 Prompt: '{prompt_text}'")
    print(f"⚙️ Config: temp={temperature}, budget={thinking_budget}, max_tokens={max_tokens}")
    print(f"📡 Calling chat_completion_stream via chat.py ...")
    print("=" * 70)

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
            # When chunk is a regular string, it is content (after </think>)
            if not has_content and has_reasoning:
                think_end_detected = True
                print("\n\n" + "-" * 35 + " [THINK END DETECTED -> CONTENT START] " + "-" * 35 + "\n")
            has_content = True
            content_chunks.append(chunk)
            print(chunk, end="", flush=True)

    t_end = time.perf_counter()
    total_duration = t_end - t_start

    full_reasoning = "".join(reasoning_chunks).strip()
    full_content = "".join(content_chunks).strip()

    print("\n" + "=" * 70)
    print("🧠 MODEL REASONING TRACE:")
    print("=" * 70)
    print(full_reasoning if full_reasoning else "(No reasoning generated)")

    print("\n" + "=" * 70)
    print("📝 MODEL CONTENT RESPONSE:")
    print("=" * 70)
    print(full_content if full_content else "(No content generated)")

    print("\n" + "=" * 70)
    print("📊 VERIFICATION RESULTS:")
    print("=" * 70)
    print(f"• TTFT:                   {ttft:.4f} s" if ttft else "• TTFT: N/A")
    print(f"• Total Duration:         {total_duration:.2f} s")
    print(f"• Reasoning Chunks Count: {len(reasoning_chunks)}")
    print(f"• Content Chunks Count:   {len(content_chunks)}")
    print(f"• Think End Detected:     {'✅ YES' if think_end_detected or (has_content and not has_reasoning) else '❌ NO'}")
    print("=" * 70)

    if think_end_detected or (has_content and full_content):
        print("✅ SUCCESS: Think end token was sent/processed and content was produced successfully!")
        return 0
    else:
        print("❌ FAILED: Think end was not detected or no content was produced.")
        return 1

if __name__ == "__main__":
    sys.exit(test_hello_think_end())
