import json
import urllib.request
import time
payload = json.dumps({
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "To read a URL, output EXACTLY the following syntax: <call_url>http://example.com</call_url> and wait for the response. Do not output anything else on that line."},
        {"role": "user", "content": "Read https://example.com"}
    ],
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": True,
    "stop": ["</call_url>"]
}).encode("utf-8")

req = urllib.request.Request(
    "http://127.0.0.1:8001/v1/chat/completions",
    data=payload,
    headers={"Content-Type": "application/json", "Accept": "text/event-stream"},
    method="POST",
)

with urllib.request.urlopen(req) as response:
    for line in response:
        line = line.decode("utf-8").strip()
        if not line or not line.startswith("data:"): continue
        if line == "data: [DONE]": break
        try:
            chunk = json.loads(line[6:])
            if "choices" in chunk and chunk["choices"]:
                delta = chunk["choices"][0].get("delta", {})
                if "content" in delta:
                    print(delta["content"], end="", flush=True)
        except Exception:
            pass
print("\nDONE")
