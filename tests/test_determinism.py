"""Determinism test: same input, temp=0 (greedy), run 3 times.
If outputs differ, there's a race condition or uninitialized memory."""
import requests, json, time, sys

url = "http://127.0.0.1:8001/v1/chat/completions"

# Wait for server
for attempt in range(30):
    try:
        r = requests.get("http://127.0.0.1:8001/v1/models", timeout=2)
        if r.status_code == 200:
            print("Server ready")
            break
    except:
        time.sleep(2)
else:
    print("Server not ready, exiting")
    sys.exit(1)

# Use the EXACT same conversation as the user's test
messages = [
    {"role": "system", "content": "You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely."},
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hello! It's great to see you. How can I assist you today? Feel free to share any questions or topics on your mind, and we'll get through them together.\n\nWhat would you like to talk about?"},
    {"role": "user", "content": "what is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome. It's a city with incredible history and, as you might know, it's also the home to one of the most famous ancient structures in the world\u2014the Colosseum! \n\nIs there anything else about Italy or its cities I can help you with?"},
    {"role": "user", "content": "what is the capital of Belgium?"},
    {"role": "assistant", "content": "The capital of Belgium is Brussels. It's also the largest city in the European Union and often considered the EU's unofficial capital because it hosts many major institutions there, including the headquarters of NATO as well.\n\nLet me know if you need more information about Belgium or its cities! I'm happy to help with any other questions you have.\n\n"},
    {"role": "user", "content": "what is commodore Amiga?"},
]

outputs = []
for run in range(3):
    resp = requests.post(url, json={
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.0,
        "stream": False
    })
    data = resp.json()
    answer = data["choices"][0]["message"]["content"]
    tokens = data["usage"]["completion_tokens"]
    outputs.append(answer)
    print(f"Run {run+1} ({tokens} tokens): {answer[:150]}")
    print()

# Check determinism
if outputs[0] == outputs[1] == outputs[2]:
    print("=" * 60)
    print("✓ DETERMINISTIC: All 3 runs produced identical output")
else:
    print("=" * 60)
    print("✗ NON-DETERMINISTIC: Outputs differ!")
    for i in range(3):
        for j in range(i+1, 3):
            if outputs[i] != outputs[j]:
                print(f"  Run {i+1} ≠ Run {j+1}")
                # Find first difference
                for k, (a, b) in enumerate(zip(outputs[i], outputs[j])):
                    if a != b:
                        print(f"  First diff at char {k}: '{outputs[i][max(0,k-20):k+20]}' vs '{outputs[j][max(0,k-20):k+20]}'")
                        break
