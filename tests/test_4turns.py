"""Reproduce user's 4-turn chat session to diagnose degradation."""
import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"
messages = []

questions = [
    "Hello",
    "what is the capital of Italy?",
    "what is the capital of Belgium?",
    "what is commodore Amiga?",
]

for q in questions:
    messages.append({"role": "user", "content": q})
    
    # Print the full prompt being sent
    print(f"--- Turn {len(messages)//2 + 1}: Sending {len(messages)} messages ---")
    
    payload = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 200,
        "temperature": 0.6,
        "repetition_penalty": 1.1,
        "stream": False
    }
    resp = requests.post(url, json=payload)
    data = resp.json()
    
    usage = data.get("usage", {})
    print(f"  Prompt tokens: {usage.get('prompt_tokens', '?')}, Completion tokens: {usage.get('completion_tokens', '?')}")
    
    answer = data["choices"][0]["message"]["content"]
    messages.append({"role": "assistant", "content": answer})
    print(f"You ❯ {q}")
    print(f"Assistant ❯ {answer}")
    print()
