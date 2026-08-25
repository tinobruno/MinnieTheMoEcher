"""Test the exact 3-question scenario: Italy, Belgium, Commodore Amiga."""
import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"
messages = []

questions = [
    "what is the capital of Italy?",
    "what is the capital of Belgium?",
    "what is the Commodore Amiga?",
]

for q in questions:
    messages.append({"role": "user", "content": q})
    payload = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.0,
        "stream": False
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": answer})
    print(f"You ❯ {q}")
    print(f"Assistant ❯ {answer}")
    print()
