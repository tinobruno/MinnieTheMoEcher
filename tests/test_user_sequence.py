import urllib.request
import json

def chat(messages):
    req = urllib.request.Request(
        "http://127.0.0.1:8001/v1/chat/completions",
        data=json.dumps({
            "model": "deepseek-v4-flash",
            "messages": messages,
            "max_tokens": 100,
            "temperature": 0.0
        }).encode(),
        headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"]

messages = []

# Turn 1: 3 fruits
print("--- TURN 1 ---")
messages.append({"role": "user", "content": "name 3 common fruits."})
ans1 = chat(messages)
print("Assistant 1:", repr(ans1))
messages.append({"role": "assistant", "content": ans1})

# Turn 2: 3 fruits again
print("\n--- TURN 2 ---")
messages.append({"role": "user", "content": "name 3 common fruits."})
ans2 = chat(messages)
print("Assistant 2:", repr(ans2))
messages.append({"role": "assistant", "content": ans2})

# Turn 3: capital of italy
print("\n--- TURN 3 ---")
messages.append({"role": "user", "content": "what is the capital of italy?"})
ans3 = chat(messages)
print("Assistant 3:", repr(ans3))
messages.append({"role": "assistant", "content": ans3})

# Turn 4: I asked what is the capital of Italy?
print("\n--- TURN 4 ---")
messages.append({"role": "user", "content": "I asked what is the capital of Italy?"})
ans4 = chat(messages)
print("Assistant 4:", repr(ans4))
messages.append({"role": "assistant", "content": ans4})

# Turn 5: capital of belgium
print("\n--- TURN 5 ---")
messages.append({"role": "user", "content": "and the capital of Belgium?"})
ans5 = chat(messages)
print("Assistant 5:", repr(ans5))

