import urllib.request
import json
req = urllib.request.Request("http://localhost:8001/v1/chat/completions",
    data=json.dumps({
        "model":"test",
        "messages":[{"role":"user","content":"hello"}],
        "stream":False,
        "max_tokens":15
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    print(json.loads(r.read())["choices"][0]["message"]["content"])
