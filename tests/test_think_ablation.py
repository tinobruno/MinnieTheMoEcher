import json
import requests

def chat(messages):
    prompt = ""
    for m in messages:
        if m["role"] == "user":
            prompt += "<｜User｜>" + m["content"]
        elif m["role"] == "assistant":
            prompt += "<｜Assistant｜>" + m["content"] + "<｜end▁of▁sentence｜>"
    prompt += "<｜User｜>Do you know what is the Commodore Amiga?<｜Assistant｜>"
    
    req = {
        "prompt": prompt,
        "max_tokens": 100,
        "temperature": 0.6,
        "repetition_penalty": 1.0,
        "stream": False
    }
    resp = requests.post("http://localhost:8001/v1/completions", json=req)
    print(resp.json()["choices"][0]["text"])

print("--- WITHOUT THINK ---")
chat([
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "Chào!"}
])

print("\n--- WITH THINK ---")
chat([
    {"role": "user", "content": "Hi"},
    {"role": "assistant", "content": "<think>\nThe user is saying hi in English, I will respond with a greeting.\n</think>\nChào!"}
])
