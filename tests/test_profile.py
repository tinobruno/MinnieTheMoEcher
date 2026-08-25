import requests
import time

url = "http://localhost:8080/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "minnie-the-moecher",
    "messages": [
        {"role": "user", "content": "Explain relativity briefly."}
    ],
    "stream": False,
    "max_tokens": 20
}

start = time.time()
response = requests.post(url, headers=headers, json=data)
print(f"Elapsed: {time.time() - start:.3f}s")
