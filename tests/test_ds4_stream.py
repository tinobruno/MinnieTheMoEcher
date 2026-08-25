import urllib.request
import json

payload = json.dumps({"model": "deepseek", "messages": [{"role": "user", "content": "Hello"}], "max_tokens": 150, "temperature": 0.6, "stream": True}).encode("utf-8")
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", data=payload)
print("Assistant: ", end="")
with urllib.request.urlopen(req) as resp:
    for line in resp:
        if b"content" in line:
            chunk = json.loads(line[6:])
            if "choices" in chunk and len(chunk["choices"]) > 0:
                print(chunk["choices"][0].get("delta", {}).get("content", ""), end="", flush=True)
print()
