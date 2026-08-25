import urllib.request
import json
import time

def send_chat(messages):
    payload = json.dumps({
        "model": "deepseek-v4-flash-0731",
        "messages": messages,
        "max_tokens": 150,
        "temperature": 0.0,
        "repetition_penalty": 1.05
    }).encode("utf-8")
    
    req = urllib.request.Request(
        "http://localhost:8001/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST"
    )
    
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())["choices"][0]["message"]["content"]

try:
    print("Testing Turn 2 directly...")
    msgs = [
        {"role": "user", "content": "What is the capital of Italy?"},
        {"role": "assistant", "content": "The capital of Italy is Rome. It is not only the capital of the country but also one of the most historically and culturally significant cities in the world, known for its ancient landmarks like the Colosseum, the Roman Forum, and the Vatican City which is a separate entity within it. Rome has been the capital since 1871, after the unification of Italy as a kingdom with the unification of the various states and territories that were part of the country at that time under the rule of the King of Italy."},
        {"role": "user", "content": "And the capital of Belgium?"}
    ]
    reply = send_chat(msgs)
    print("Reply:", reply)
except Exception as e:
    print("Error:", e)
