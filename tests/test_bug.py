import urllib.request
import json

def chat(messages, rep_penalty):
    payload = json.dumps({
        "model": "deepseek-v4-flash-0731",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.6,
        "repetition_penalty": rep_penalty
    }).encode("utf-8")
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())['choices'][0]['message']['content']

m = [
    {"role": "system", "content": "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely."},
    {"role": "user", "content": "What is the capital of Italy ?"}
]
r1 = chat(m, 1.1)
print("Turn 1:", r1)
m.append({"role": "assistant", "content": r1})
m.append({"role": "user", "content": "And the capital of Belgium ?"})
r2 = chat(m, 1.1)
print("Turn 2:", r2)
