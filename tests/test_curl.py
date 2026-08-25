import urllib.request
import json

url = "http://localhost:8001/v1/chat/completions"
payload = json.dumps({
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "Hello"}],
    "max_tokens": 100,
    "temperature": 0.6,
    "stream": False
}).encode("utf-8")

req = urllib.request.Request(
    url,
    data=payload,
    headers={"Content-Type": "application/json"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=600) as resp:
        print(resp.read().decode('utf-8'))
except Exception as e:
    print(e)
