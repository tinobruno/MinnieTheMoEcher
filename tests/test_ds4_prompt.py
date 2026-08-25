import urllib.request
import json
payload = json.dumps({"model": "deepseek", "messages": [{"role": "system", "content": "You are a helpful, concise AI assistant."}, {"role": "user", "content": "Hello"}], "max_tokens": 200, "temperature": 0.6, "stream": True}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=payload)
with urllib.request.urlopen(req) as resp:
    print(resp.read().decode("utf-8"))
