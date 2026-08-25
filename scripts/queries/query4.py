import requests
import json

url = "http://localhost:8001/v1/chat/completions"
headers = {"Content-Type": "application/json"}
data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "You have access to the following functions:\n[\n  {\"name\": \"fetch_url\", \"description\": \"Fetch URL\"}\n]\nTo invoke a function, output a JSON array of tool calls inside a <tool_call> block.\n\nUser: fetch the URL https://example.com"}
    ],
    "stream": False
}

response = requests.post(url, headers=headers, data=json.dumps(data))
print(response.json())
