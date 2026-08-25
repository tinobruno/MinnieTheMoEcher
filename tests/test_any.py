import urllib.request
import json
url = "http://127.0.0.1:8001/v1/chat/completions"
def chat(prompt):
    payload = json.dumps({"model": "deepseek-v4-flash", "messages": [{"role": "user", "content": prompt}], "max_tokens": 128, "temperature": 0.0, "stream": False}).encode("utf-8")
    req = urllib.request.Request(url, data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode("utf-8"))["choices"][0]["message"]["content"]
print(chat("Write a python script to reverse a string."))
