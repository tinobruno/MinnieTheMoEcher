import urllib.request
import json
import time

prompt = "Hello! " * 50 + "What is 2+2?"
messages = [{"role": "user", "content": prompt}]
req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", method="POST", headers={"Content-Type": "application/json"})
payload = json.dumps({"model": "deepseek-v4-flash", "messages": messages, "max_tokens": 20, "temperature": 0.0, "stream": False}).encode("utf-8")
try:
    with urllib.request.urlopen(req, data=payload) as response:
        print(json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"])
except Exception as e:
    print(e)
