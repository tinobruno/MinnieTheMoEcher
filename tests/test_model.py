import requests
import json
import time

def chat():
    url = "http://127.0.0.1:8080/v1/chat/completions"
    headers = {"Content-Type": "application/json"}
    data = {
        "model": "deepseek",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant. Always use <think> and </think> tags to show your reasoning before answering."},
            {"role": "user", "content": "Hello"}
        ],
        "stream": True,
        "max_tokens": 100
    }
    
    start = time.time()
    try:
        response = requests.post(url, headers=headers, json=data, stream=True)
        for line in response.iter_lines():
            if line:
                print(line.decode('utf-8'))
    except Exception as e:
        print("Error:", e)

chat()
