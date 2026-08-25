"""Print full responses to see exactly WHERE bleeding occurs."""
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

resp = requests.post(url, json={
    "model": "deepseek-v4-flash",
    "messages": messages,
    "max_tokens": 300,
    "temperature": 0.6,
    "stream": False
})
data = resp.json()
print(f"Tokens: prompt={data['usage']['prompt_tokens']}, completion={data['usage']['completion_tokens']}")
print(f"\nFull answer:\n{data['choices'][0]['message']['content']}")
