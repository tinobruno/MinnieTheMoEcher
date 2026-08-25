"""Statistical test: run Turn 4 (Amiga) 10 times at temp=0.6 to measure bleeding rate."""
import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"

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

clean = 0
bleed = 0
for run in range(10):
    resp = requests.post(url, json={
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.6,
        "repetition_penalty": 1.1,
        "stream": False
    })
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    has_bleed = any(w in answer.lower() for w in bleed_words)
    if has_bleed:
        bleed += 1
        print(f"Run {run+1}: ✗ BLEED | {answer[:100]}...")
    else:
        clean += 1
        print(f"Run {run+1}: ✓ CLEAN | {answer[:100]}...")

print(f"\n{'='*60}")
print(f"RESULTS: {clean}/10 clean ({clean*10}%), {bleed}/10 bleeding ({bleed*10}%)")
