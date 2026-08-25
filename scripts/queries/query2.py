import requests
import json

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "fetch the URL https://example.com"}
    ],
    "stream": False,
    "tools": [{
        "type": "function",
        "function": {
            "name": "fetch_url",
            "description": "Fetch URL",
            "parameters": {
                "type": "object",
                "properties": {
                    "url": {"type": "string"}
                }
            }
        }
    }]
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
