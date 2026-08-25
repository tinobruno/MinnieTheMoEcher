import urllib.request
import json
req = urllib.request.Request("http://localhost:8002/v1/chat/completions",
    data=json.dumps({"model":"test","messages":[{"role":"user","content":"hello"}],"stream":False,"max_tokens":10}).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    print(r.read().decode())
