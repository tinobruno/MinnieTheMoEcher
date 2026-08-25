import urllib.request
import json
url = "http://127.0.0.1:8001/v1/chat/completions"
def chat(messages):
    payload = json.dumps({"model": "deepseek-v4-flash", "messages": messages, "max_tokens": 128, "temperature": 0.0, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
m = [{"role": "user", "content": "Hello, this is a slightly longer sentence to see if it breaks."}]
print(repr(chat(m)))
