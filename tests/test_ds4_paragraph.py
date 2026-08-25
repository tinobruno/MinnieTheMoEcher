import urllib.request
import json

default_system = "You are a helpful and concise AI assistant. You can call tools if needed. To read a URL, use exactly: <call_url>http://example.com</call_url>. Answer the user directly without meta-commentary."

def generate(prompt):
    messages = [{"role": "system", "content": default_system}, {"role": "user", "content": prompt}]
    payload = json.dumps({"model": "deepseek", "messages": messages, "max_tokens": 100, "temperature": 0.0}).encode("utf-8")
    req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=payload)
    with urllib.request.urlopen(req) as resp:
        for line in resp:
            if b"content" in line:
                print(line.decode().strip())

generate("Hello")
