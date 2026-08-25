import json
import urllib.request

url = "http://127.0.0.1:8001/v1/chat/completions"

def chat(messages, max_tokens=150):
    payload = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": 0.0,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]

print("=== TEST 1: Single Turn (Fruits) ===")
msg1 = [{"role": "user", "content": "Name three common fruits."}]
reply1 = chat(msg1)
print("Q: Name three common fruits.")
print("A:", reply1)
print("-" * 50)

print("\n=== TEST 2: Multi-turn Conversation ===")
history = [
    {"role": "user", "content": "What is the capital of Italy?"},
]
reply_it = chat(history)
print("User: What is the capital of Italy?")
print("Assistant:", reply_it)

history.append({"role": "assistant", "content": reply_it})
history.append({"role": "user", "content": "What is its population?"})

reply_pop = chat(history)
print("User: What is its population?")
print("Assistant:", reply_pop)

history.append({"role": "assistant", "content": reply_pop})
history.append({"role": "user", "content": "Name two major landmarks in that city."})

reply_landmarks = chat(history)
print("User: Name two major landmarks in that city.")
print("Assistant:", reply_landmarks)
print("-" * 50)
