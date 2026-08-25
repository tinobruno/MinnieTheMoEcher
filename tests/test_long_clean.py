import requests

prompt = "The Quick Brown Fox Jumps Over The Lazy Dog. " * 10
prompt += "The sun rises in the east and sets in the west. " * 5
prompt += "Water boils at 100 degrees Celsius at sea level. " * 5
prompt += "What is the capital of Italy?"

response = requests.post(
    "http://127.0.0.1:8001/v1/chat/completions",
    json={
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 10,
        "temperature": 0.0
    }
)
print(response.json())
