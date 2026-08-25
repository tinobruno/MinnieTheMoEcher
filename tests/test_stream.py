import requests
import json
data = {
    "model": "deepseek-v4-flash",
    "messages": [{"role": "user", "content": "what is amiga?"}],
    "stream": True
}
print("Testing with current server...")
try:
    response = requests.post("http://localhost:8001/v1/chat/completions", json=data, stream=True, timeout=30)
    for line in response.iter_lines():
        if line:
            print(line.decode('utf-8'))
except Exception as e:
    print(e)
