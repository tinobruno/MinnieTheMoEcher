import urllib.request
import json

def test_len(n):
    prompt = "Hello " * n
    messages = [{"role": "user", "content": prompt}]
    req = urllib.request.Request("http://127.0.0.1:8001/v1/chat/completions", method="POST", headers={"Content-Type": "application/json"})
    payload = json.dumps({"model": "deepseek-v4-flash", "messages": messages, "max_tokens": 10, "temperature": 0.0, "stream": False}).encode("utf-8")
    with urllib.request.urlopen(req, data=payload) as response:
        return json.loads(response.read().decode("utf-8"))["choices"][0]["message"]["content"]

for i in range(1, 20):
    print(f"Len {i}: {repr(test_len(i))}")
