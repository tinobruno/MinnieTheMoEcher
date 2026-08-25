import json
import urllib.request

url = "http://localhost:8001/v1/chat/completions"
messages = [{"role": "user", "content": "hello"}]
payload = json.dumps({"model": "deepseek-v4-flash", "messages": messages, "stream": True}).encode("utf-8")
req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as resp:
    for line in resp:
        if line.strip():
            print(repr(line.decode('utf-8')))
