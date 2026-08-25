import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat import chat_completion_stream

prompt = "create a script to generate first 4 decimals of PI"
print(f"Testing prompt: {prompt}")

messages = [
    {"role": "system", "content": "You are a helpful assistant"},
    {"role": "user", "content": prompt}
]

reasoning = []
content = []
start = time.time()
for chunk in chat_completion_stream(
    "http://localhost:8001",
    messages,
    max_tokens=1024,
    temperature=0.6,
    thinking="enabled",
    thinking_budget=1024
):
    if isinstance(chunk, dict) and chunk.get("type") == "reasoning":
        reasoning.append(chunk["content"])
        sys.stdout.write(chunk["content"])
        sys.stdout.flush()
    elif isinstance(chunk, str):
        content.append(chunk)
        sys.stdout.write(chunk)
        sys.stdout.flush()

print("\n\n" + "="*50)
print(f"Total time: {time.time() - start:.2f}s")
print(f"Reasoning length: {len(''.join(reasoning))} chars")
print(f"Content length: {len(''.join(content))} chars")
print("Content snippet (last 300 chars):", repr("".join(content)[-300:]))
