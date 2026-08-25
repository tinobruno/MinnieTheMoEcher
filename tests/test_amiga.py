import os
import sys
import json
import time
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from chat import chat_completion_stream

prompt = "tell me about commodore Amiga, in particular about Amiga 3000 and Amiga 4000, and their UNIX variants."
print(f"Testing prompt: {prompt}\n")

messages = [
    {"role": "user", "content": prompt}
]

reasoning = []
content = []
start = time.time()
for chunk in chat_completion_stream(
    "http://localhost:8001",
    messages,
    max_tokens=1024,
    temperature=1.0,
    thinking="enabled",
    thinking_budget=2048
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
