import urllib.request
import json
req = urllib.request.Request("http://localhost:8001/v1/chat/completions",
    data=json.dumps({
        "model":"test",
        "messages":[
            {"role":"system","content":"You are a helpful, concise AI assistant. Answer questions directly after your thoughts. "},
            {"role":"user","content":"hello"}
        ],
        "stream":True,
        "max_tokens":100
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    for line in r:
        if b"reasoning_content" in line or b"content" in line:
            pass # ignore
        elif b"finish_reason" in line:
            print(line.decode().strip())
        else:
            if line.strip(): print("Raw:", line.decode().strip())
