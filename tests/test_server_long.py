import requests

url = "http://127.0.0.1:8080/v1/chat/completions"
data = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "Write a short 3 paragraph story about a robot learning to paint."}
    ],
    "max_tokens": 150,
    "temperature": 0.6
}
response = requests.post(url, json=data)
print(response.json()['choices'][0]['message']['content'])
