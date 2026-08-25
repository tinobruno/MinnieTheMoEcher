import requests, json

url = 'http://127.0.0.1:8001/v1/chat/completions'
data = {
    'messages': [
        {'role': 'system', 'content': 'You are a helpful, friendly, and knowledgeable AI assistant. Answer clearly and concisely.'},
        {'role': 'user', 'content': 'What is the capital of Italy?'},
        {'role': 'assistant', 'content': 'The capital of Italy is Rome.'},
        {'role': 'user', 'content': 'And what is the capital of Belgium?'}
    ],
    'stream': True
}

response = requests.post(url, json=data, stream=True)
for line in response.iter_lines():
        line_str = line.decode('utf-8')
        if line_str.startswith("data: "):
            data_str = line_str[6:]
            if data_str == "[DONE]":
                break
            try:
                chunk = json.loads(data_str)
                content = chunk.get("choices", [{}])[0].get("delta", {}).get("content", "")
                if content:
                    print(content, end="", flush=True)
            except json.JSONDecodeError:
                pass
print()
