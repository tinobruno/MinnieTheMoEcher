import urllib.request
import json
req = urllib.request.Request("http://localhost:8001/v1/chat/completions",
    data=json.dumps({
        "model":"test",
        "messages":[{"role":"user","content":"hello\n<think>\n"}],
        "stream":True,
        "max_tokens":100
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    for line in r:
        if b"content" in line or b"reasoning_content" in line:
            print(line.decode().strip())
