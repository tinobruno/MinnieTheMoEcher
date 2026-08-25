import requests, json

url = 'http://127.0.0.1:8001/v1/chat/completions'
data = {
    'messages': [
        {'role': 'user', 'content': 'What is the capital of Italy?'},
        {'role': 'assistant', 'content': 'The capital of Italy is Rome.'},
        {'role': 'user', 'content': 'And what is the capital of Belgium?'}
    ]
}

response = requests.post(url, json=data)
print(response.text)
