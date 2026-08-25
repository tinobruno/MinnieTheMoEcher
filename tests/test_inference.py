import requests
import json

response = requests.post("http://localhost:8001/chat/completions", json={
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France? Please answer clearly."}
    ],
    "max_tokens": 100,
    "temperature": 0.0,
    "stream": False
})

print(response.json())
