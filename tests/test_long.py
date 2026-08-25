import requests, json
prompt = "Explain the core concepts of quantum computing in one short paragraph."
req = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": prompt}
    ],
    "max_tokens": 100,
    "temperature": 0.7
}
r = requests.post("http://localhost:8001/v1/chat/completions", json=req)
print(r.json()['choices'][0]['message']['content'])
