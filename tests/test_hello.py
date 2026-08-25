import requests, json
resp = requests.post("http://localhost:8001/v1/chat/completions", json={
    "model": "moecher",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.7,
    "max_tokens": 100
})
data = resp.json()
print("Reasoning:", data["choices"][0]["message"].get("reasoning_content", ""))
print("Content:", data["choices"][0]["message"].get("content", ""))
