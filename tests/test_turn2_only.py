import requests
import json

def chat(messages):
    resp = requests.post("http://localhost:8001/v1/chat/completions", 
                         json={"messages": messages, "temperature": 0.0})
    if resp.status_code == 200:
        return resp.json()["choices"][0]["message"]["content"]
    else:
        print("Error:", resp.text)
        return ""

hist = [
    {"role": "user", "content": "What is the capital of Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin.\n\nIt's the United Kingdom of which?"},
    {"role": "user", "content": "Where is it located?"}
]

r = chat(hist)
print("Turn 2 A:", repr(r))
