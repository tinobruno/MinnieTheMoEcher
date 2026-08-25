import requests

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of Italy?"}
]

res = requests.post("http://localhost:8001/v1/chat/completions", json={
    "model": "moecher",
    "messages": messages,
    "max_tokens": 100,
    "repetition_penalty": 1.0,
    "temperature": 0.6
})
r1 = res.json()
msg1 = r1["choices"][0]["message"]["content"]
print("Turn 1:", msg1)

messages.append({"role": "assistant", "content": msg1})
messages.append({"role": "user", "content": "What is its population?"})

res2 = requests.post("http://localhost:8001/v1/chat/completions", json={
    "model": "moecher",
    "messages": messages,
    "max_tokens": 100,
    "repetition_penalty": 1.0,
    "temperature": 0.6
})
print("Turn 2:", res2.json()["choices"][0]["message"]["content"])
