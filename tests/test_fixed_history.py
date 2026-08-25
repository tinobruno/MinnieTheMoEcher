"""Test with FIXED short assistant responses but temperature=0.6.
This isolates whether the issue is the verbose history or the sampling."""
import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"

# Use fixed, concise assistant responses (like greedy test)
messages = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "what is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome."},
    {"role": "user", "content": "what is the capital of Belgium?"},
    {"role": "assistant", "content": "The capital of Belgium is Brussels."},
    {"role": "user", "content": "what is commodore Amiga?"},
]

print("="*60)
print("TEST: Fixed short history, temperature=0.6 (3 runs)")
print("="*60)

for run in range(3):
    resp = requests.post(url, json={
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.6,
        "stream": False
    })
    data = resp.json()
    answer = data['choices'][0]['message']['content']
    # Check for bleeding
    has_italy = "italy" in answer.lower() or "rome" in answer.lower() or "roma" in answer.lower()
    has_belgium = "belgium" in answer.lower() or "brussels" in answer.lower()
    has_confusion = "confusion" in answer.lower() or "mistake" in answer.lower() or "clarif" in answer.lower()
    status = "✓ CLEAN" if not (has_italy or has_belgium or has_confusion) else "✗ BLEEDING"
    print(f"\nRun {run+1}: {status}")
    print(f"  Answer: {answer[:200]}")
    if has_italy: print(f"  ⚠ Contains Italy/Rome reference")
    if has_belgium: print(f"  ⚠ Contains Belgium/Brussels reference")
    if has_confusion: print(f"  ⚠ Contains confusion/mistake/clarify")
