import requests
import json
import time

payload = {
    "model": "deepseek-v4",
    "messages": [
        {"role": "user", "content": "play triquila - tony pitony"}
    ],
    "tools": "default"
}

# Call /api/kv/reset
print("Resetting KV cache...")
requests.post("http://localhost:8001/api/kv/reset")
time.sleep(1)

# Send the request
print("Sending chat request...")
resp = requests.post("http://localhost:8001/v1/chat/completions", json=payload, stream=True)
for line in resp.iter_lines():
    if line:
        print(line.decode('utf-8'))
