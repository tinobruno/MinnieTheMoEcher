import requests

def chat(content):
    res = requests.post("http://127.0.0.1:8001/v1/chat/completions", json={
        "model": "deepseek-v4-flash-0731",
        "messages": [{"role": "user", "content": content}],
        "max_tokens": 100
    })
    if res.status_code == 200:
        print(res.json()["choices"][0]["message"]["content"])
    else:
        print(f"Error {res.status_code}: {res.text}")

long_text = "The quick brown fox jumps over the lazy dog. " * 50
long_text += " What is the animal that jumps?"
chat(long_text)
