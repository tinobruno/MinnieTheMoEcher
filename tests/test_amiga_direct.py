"""Test with temperature=0 (greedy) to eliminate sampling noise.
Also test with system prompt (like chat.py) and without."""
import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"

sys_prompt = "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely."

# Test 1: With system prompt, temp=0 (greedy)
print("="*60)
print("TEST 1: With system prompt, temperature=0.0 (greedy)")
print("="*60)
messages1 = [
    {"role": "system", "content": sys_prompt},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "what is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome."},
    {"role": "user", "content": "what is the capital of Belgium?"},
    {"role": "assistant", "content": "The capital of Belgium is Brussels."},
    {"role": "user", "content": "what is commodore Amiga?"},
]
resp = requests.post(url, json={
    "model": "deepseek-v4-flash",
    "messages": messages1,
    "max_tokens": 300,
    "temperature": 0.0,
    "stream": False
})
data = resp.json()
print(f"Tokens: prompt={data['usage']['prompt_tokens']}, completion={data['usage']['completion_tokens']}")
print(f"Answer:\n{data['choices'][0]['message']['content']}\n")

# Test 2: WITHOUT system prompt, temp=0
print("="*60)
print("TEST 2: No system prompt, temperature=0.0 (greedy)")
print("="*60)
messages2 = [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! How can I assist you today?"},
    {"role": "user", "content": "what is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome."},
    {"role": "user", "content": "what is the capital of Belgium?"},
    {"role": "assistant", "content": "The capital of Belgium is Brussels."},
    {"role": "user", "content": "what is commodore Amiga?"},
]
resp = requests.post(url, json={
    "model": "deepseek-v4-flash",
    "messages": messages2,
    "max_tokens": 300,
    "temperature": 0.0,
    "stream": False
})
data = resp.json()
print(f"Tokens: prompt={data['usage']['prompt_tokens']}, completion={data['usage']['completion_tokens']}")
print(f"Answer:\n{data['choices'][0]['message']['content']}\n")

# Test 3: Amiga ONLY, no history, temp=0
print("="*60)
print("TEST 3: Amiga only (no conversation history), temperature=0.0")
print("="*60)
messages3 = [
    {"role": "user", "content": "what is commodore Amiga?"},
]
resp = requests.post(url, json={
    "model": "deepseek-v4-flash",
    "messages": messages3,
    "max_tokens": 300,
    "temperature": 0.0,
    "stream": False
})
data = resp.json()
print(f"Tokens: prompt={data['usage']['prompt_tokens']}, completion={data['usage']['completion_tokens']}")
print(f"Answer:\n{data['choices'][0]['message']['content']}\n")
