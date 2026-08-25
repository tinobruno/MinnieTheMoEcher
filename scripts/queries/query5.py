import requests
import json

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "Hello, how are you?"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
