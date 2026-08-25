import urllib.request
import json

payload = {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "hello"}],
    "max_tokens": 50,
    "temperature": 0.6,
    "stream": True
}
req = urllib.request.Request(
    "http://localhost:8001/v1/chat/completions",
    data=json.dumps(payload).encode("utf-8"),
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    method="POST",
)

try:
    with urllib.request.urlopen(req, timeout=10) as resp:
        for line in resp:
            print(repr(line))
except Exception as e:
    print("Error:", e)
