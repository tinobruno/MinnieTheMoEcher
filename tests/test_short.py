import requests

response = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": "Hello"}],
        "max_tokens": 5,
        "temperature": 0.0
    }
)
print(response.json())
