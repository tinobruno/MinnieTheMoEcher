import requests
import json
import time

print("Sending request 1 (will abort mid-stream)...")
try:
    response = requests.post(
        "http://localhost:8001/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Write a long essay."}], "stream": True, "max_tokens": 100},
        stream=True
    )
    for i, line in enumerate(response.iter_lines()):
        if line:
            print("Received:", line.decode("utf-8"))
            if i > 5:
                print("Aborting stream...")
                response.close() # simulate client disconnect
                break
except Exception as e:
    print("Error:", e)

print("Waiting a moment...")
time.sleep(2)

print("Sending request 2 (should succeed)...")
try:
    response = requests.post(
        "http://localhost:8001/v1/chat/completions",
        json={"messages": [{"role": "user", "content": "Hello!"}], "stream": True, "max_tokens": 10},
        stream=True
    )
    for line in response.iter_lines():
        if line:
            print("Received 2:", line.decode("utf-8"))
except Exception as e:
    print("Error 2:", e)
