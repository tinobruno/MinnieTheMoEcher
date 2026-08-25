"""Test Turn 4 (Amiga) at multiple temperatures with fixed short history.
Find the max temperature that still produces clean output."""
import requests, json, time

url = "http://127.0.0.1:8001/v1/chat/completions"

# Wait for server
for _ in range(10):
    try:
        requests.get(f"{url.rsplit('/',2)[0]}/v1/models", timeout=2)
        break
    except: time.sleep(3)

messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "what is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome."},
    {"role": "user", "content": "what is the capital of Belgium?"},
    {"role": "assistant", "content": "The capital of Belgium is Brussels."},
    {"role": "user", "content": "what is commodore Amiga?"},
]

bleed_words = ["italy", "rome", "roma", "belgium", "brussels", "confusion", 
               "mistake", "clarif", "capital", "previous"]

for temp in [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6]:
    resp = requests.post(url, json={
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 200,
        "temperature": temp,
        "repetition_penalty": 1.1,
        "stream": False
    })
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    has_bleed = any(w in answer.lower() for w in bleed_words)
    status = "✗ BLEED" if has_bleed else "✓ CLEAN"
    print(f"temp={temp:.1f}: {status} | {answer[:120]}...")
