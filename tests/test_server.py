import requests

url = "http://127.0.0.1:8080/v1/chat/completions"
data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "What is the capital of Italy?"}
    ],
    "max_tokens": 128,
    "temperature": 0.6
}
response = requests.post(url, json=data)
print(response.json()['choices'][0]['message']['content'])
