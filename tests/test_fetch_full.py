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
        "max_tokens":200
    }).encode("utf-8"),
    headers={"Content-Type": "application/json"})
with urllib.request.urlopen(req) as r:
    reasoning = ""
    content = ""
    for line in r:
        line = line.decode().strip()
        if line.startswith("data: {"):
            d = json.loads(line[6:])
            if "choices" in d and len(d["choices"])>0:
                delta = d["choices"][0].get("delta", {})
                if "reasoning_content" in delta:
                    reasoning += delta["reasoning_content"]
                if "content" in delta:
                    content += delta["content"]
    print("REASONING:", repr(reasoning))
    print("CONTENT:", repr(content))
