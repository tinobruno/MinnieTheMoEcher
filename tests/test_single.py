import urllib.request
import json
messages = [{"role": "user", "content": "What is 2+2?"}]
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", method="POST", headers={"Content-Type": "application/json"})
payload = json.dumps({"model": "deepseek-v4-flash", "messages": messages, "max_tokens": 100, "temperature": 0.0, "stream": False}).encode("utf-8")
with urllib.request.urlopen(req, data=payload) as response:
    print(json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"])
