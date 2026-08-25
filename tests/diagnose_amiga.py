import urllib.request
import json
import sys

prompt = "tell me about commodore Amiga, in particular about Amiga 3000 and Amiga 4000, and their UNIX variants."
messages = [{"role": "user", "content": prompt}]

payload = json.dumps({
    "model": "deepseek-v4-flash",
    "messages": messages,
    "max_tokens": 512,
    "temperature": 0.0,
    "thinking": {"type": "enabled", "budget_tokens": 4096},
    "stream": True
}).encode("utf-8")

req = urllib.request.Request(
    "http://localhost:8001/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json"}
)

print("Connecting...")
with urllib.request.urlopen(req, timeout=300) as resp:
    for line in resp:
        if line.startswith(b"data: "):
            chunk_str = line[6:].decode("utf-8").strip()
            if chunk_str == "[DONE]":
                break
            chunk = json.loads(chunk_str)
            delta = chunk["choices"][0]["delta"]
            if "reasoning_content" in delta:
                sys.stdout.write("\033[90m" + delta["reasoning_content"] + "\033[0m")
                sys.stdout.flush()
            if "content" in delta:
                sys.stdout.write(delta["content"])
                sys.stdout.flush()
print()
