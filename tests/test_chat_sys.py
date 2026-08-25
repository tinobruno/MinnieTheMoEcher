import json, urllib.request

data = json.dumps({
    "model": "moecher",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant. Always wrap your reasoning inside <think> and </think> tags."},
        {"role": "user", "content": "what is Amiga?"}
    ],
    "stream": True
}).encode()

req = urllib.request.Request("http://localhost:8001/v1/chat/completions", data=data, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
for line in resp:
    print(line.decode().strip())
