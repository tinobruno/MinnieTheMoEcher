import requests
import json
import time
import sys

API_BASE = "http://127.0.0.1:8080"

# Wait for server to be up
def wait_for_server():
    for _ in range(400):
        try:
            r = requests.get(f"{API_BASE}/")
            if r.status_code == 200:
                print("Server is up!")
                return
        except:
            time.sleep(1)
    print("Server not responding")
    sys.exit(1)

def reset_kv():
    requests.post(f"{API_BASE}/api/kv/reset")

def chat_complete(message, use_tools=True):
    payload = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant"},
            {"role": "user", "content": message}
        ],
        "temperature": 0.0,
        "max_tokens": 200,
        "stream": False
    }
    if use_tools:
        payload["tools"] = "default"
        
    r = requests.post(f"{API_BASE}/v1/chat/completions", json=payload)
    if r.status_code == 200:
        return r.json()
    else:
        return {"error": r.status_code, "msg": r.text}

wait_for_server()

print("Testing fresh start...")
res1 = chat_complete("play triquila - tony pitony")
print("Response 1:", res1.get('choices', [{}])[0].get('message', {}).get('content', res1))

print("\nResetting KV cache...")
reset_kv()

print("\nTesting post-reset...")
res2 = chat_complete("play triquila - tony pitony")
print("Response 2:", res2.get('choices', [{}])[0].get('message', {}).get('content', res2))
