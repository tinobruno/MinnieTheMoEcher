import urllib.request
import json
import time

payload = json.dumps({
    "model": "deepseek-v4-flash-0731",
    "messages": [{"role": "user", "content": "What is the capital of Italy?"}],
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": True
}).encode('utf-8')

req = urllib.request.Request("http://localhost:8001/v1/chat/completions", data=payload, headers={"Content-Type": "application/json", "Accept": "text/event-stream"})
with urllib.request.urlopen(req) as resp:
    while True:
        line = resp.readline()
        if not line: break
        print(line.decode('utf-8'), end="")
