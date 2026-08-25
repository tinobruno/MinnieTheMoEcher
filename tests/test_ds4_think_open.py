import urllib.request
import json

default_system = "You are a helpful, concise AI assistant."

def generate(prompt):
    messages = [{"role": "system", "content": default_system}, {"role": "user", "content": prompt}]
    payload = json.dumps({"model": "deepseek", "messages": messages, "max_tokens": 150, "temperature": 0.6}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=payload)
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if b"content" in line:
                chunk = json.loads(line)
                if "choices" in chunk and len(chunk["choices"]) > 0:
                    reply = chunk["choices"][0].get("delta", {}).get("content", "")
                    if reply: print(reply, end="", flush=True)

generate("What is 9 * 8?")
