#!/usr/bin/env python3
import urllib.request
import json
import time
import sys

URL = "http://localhost:8001/v1/chat/completions"
PROMPT = "Write an essay of 300 words about quantum physics"

payload = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant"},
        {"role": "user", "content": PROMPT}
    ],
    "max_tokens": 1024,
    "temperature": 0.6,
    "stream": True,
    "thinking": {"type": "enabled", "budget_tokens": 2048}
}

print("=" * 70)
print(f"🚀 Prompt: \"{PROMPT}\"")
print(f"📡 Connecting to {URL} (Streaming Mode)...")
print("=" * 70)

req = urllib.request.Request(
    URL,
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json"}
)

t_start = time.time()
t_first_token = None
tokens_generated = 0
reasoning_tokens = 0
content_tokens = 0
in_thinking = False

reasoning_buf = ""
content_buf = ""

print("\n--- [LIVE STREAMING OUTPUT] ---\n")

try:
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            line = line.decode("utf-8").strip()
            if not line or not line.startswith("data: "):
                continue
            if line == "data: [DONE]":
                break

            if t_first_token is None:
                t_first_token = time.time()

            tokens_generated += 1
            chunk = json.loads(line[6:])
            choices = chunk.get("choices", [])
            if not choices:
                continue

            delta = choices[0].get("delta", {})
            
            # Streaming reasoning tokens
            if "reasoning_content" in delta and delta["reasoning_content"]:
                if not in_thinking:
                    sys.stdout.write("\033[90m[THINKING]: ")
                    in_thinking = True
                txt = delta["reasoning_content"]
                reasoning_buf += txt
                reasoning_tokens += 1
                sys.stdout.write(txt)
                sys.stdout.flush()

            # Streaming content tokens
            elif "content" in delta and delta["content"]:
                if in_thinking:
                    sys.stdout.write("\033[0m\n\n[CONTENT]: ")
                    in_thinking = False
                txt = delta["content"]
                content_buf += txt
                content_tokens += 1
                sys.stdout.write(txt)
                sys.stdout.flush()

    if in_thinking:
        sys.stdout.write("\033[0m\n")

except KeyboardInterrupt:
    print("\n\n[Generation stopped by user]")
except Exception as e:
    print(f"\n❌ Error during generation: {e}")

t_end = time.time()
total_time = t_end - t_start
ttft = (t_first_token - t_start) if t_first_token else 0.0
decode_time = (t_end - t_first_token) if t_first_token else total_time
decode_speed = (tokens_generated / decode_time) if decode_time > 0 else 0.0

word_count = len(content_buf.split())

print("\n\n" + "=" * 70)
print("🧠 MODEL REASONING TRACE")
print("=" * 70)
print(reasoning_buf.strip() if reasoning_buf.strip() else "(No reasoning tokens)")

print("\n" + "=" * 70)
print("📝 FINAL MODEL ESSAY OUTPUT")
print("=" * 70)
print(content_buf.strip() if content_buf.strip() else "(No content generated)")

print("\n" + "=" * 70)
print("📊 PERFORMANCE & QUALITY BREAKDOWN")
print("=" * 70)
print(f"Time to First Token (TTFT): {ttft:.3f} s")
print(f"Decode Duration:            {decode_time:.3f} s")
print(f"Total Elapsed Time:         {total_time:.3f} s")
print(f"Reasoning Tokens:           {reasoning_tokens}")
print(f"Content Tokens:             {content_tokens}")
print(f"Total Tokens Generated:     {tokens_generated}")
print(f"Final Essay Word Count:     {word_count} words")
print(f"⚡ Live Decode Throughput:   {decode_speed:.2f} tok/s")
print("=" * 70)
