import urllib.request
import json

def send_chat(messages):
    payload = json.dumps({
        "model": "deepseek-v4-flash-0731",
        "messages": messages,
        "max_tokens": 50,
        "temperature": 0.0,
        "repetition_penalty": 1.05
    }).encode("utf-8")
    req = urllib.request.Request(
        "http://localhost:8001/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"]

try:
    msgs = [{"role": "user", "content": "What is the capital of Italy?"}]
    print("Run 1:", send_chat(msgs))
    print("Run 2:", send_chat(msgs))
except Exception as e:
    print("Error:", e)
