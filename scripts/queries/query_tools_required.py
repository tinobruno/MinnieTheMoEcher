import requests
import json
import sys

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "What's the weather in Tokyo?"}
    ],
    "tools": [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "location": {"type": "string"}
                    },
                    "required": ["location"]
                }
            }
        }
    ],
    "tool_choice": "required", "stream": True,
    "max_tokens": 1000
}

response = requests.post(url, headers=headers, data=json.dumps(data), stream=True)
for line in response.iter_lines():
    if line:
        print(line.decode('utf-8'))
