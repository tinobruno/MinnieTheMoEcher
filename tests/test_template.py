import urllib.request
import json

payload = json.dumps({
    "model": "deepseek",
    "messages": [
        {"role": "user", "content": "What is the capital of Italy?"},
        {"role": "assistant", "content": "Rome."},
        {"role": "user", "content": "And Belgium?"}
    ]
}).encode("utf-8")

req = urllib.request.Request("http://localhost:8001/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"})
urllib.request.urlopen(req)
