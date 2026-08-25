import urllib.request
import json

payload = json.dumps({
    "model": "deepseek",
    "messages": [
        {"role": "user", "content": "What is the capital of Italy?"},
    ]
}).encode("utf-8")

req = urllib.request.Request("http://localhost:8001/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
print(json.loads(urllib.request.urlopen(req).read().decode())["choices"][0]["message"]["content"])
