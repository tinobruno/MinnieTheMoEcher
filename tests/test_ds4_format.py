import urllib.request
import json

def generate(prompt):
    messages = [{"role": "system", "content": "You are a helpful assistant."}, {"role": "user", "content": prompt}]
    payload = json.dumps({"model": "deepseek", "messages": messages, "max_tokens": 50, "temperature": 0.0}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=payload)
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if b"content" in line:
                print(line.decode().strip())

generate("Hello!")
