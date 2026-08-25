import requests

url = "http://127.0.0.1:8080/v1/chat/completions"

def req(messages):
    data = {
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.6
    }
    response = requests.post(url, json=data)
    print("Response:", response.json()['choices'][0]['message']['content'])

print("Request 2:")
req([
    {"role": "user", "content": "what is the capital of Italy ?"},
    {"role": "assistant", "content": "You're thinking of a classic geography question! The capital of Italy is Rome (Roma in Italian).\n\nTo give you a bit more: Rome is not just the capital of Italy, but also the capital of the Lazio region. It's one of the most historically significant cities in the"},
    {"role": "user", "content": "what is the capital of Italy ?"}
])
