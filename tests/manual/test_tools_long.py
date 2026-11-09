import requests
import json
import sys

url = "http://127.0.0.1:8080/v1/chat/completions"
payload = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "play triquila - tony pitony"}
    ],
    "tools": "default",
    "max_tokens": 1024,
    "temperature": 1.0,
    "stream": True
}

response = requests.post(url, json=payload, stream=True)
for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith('data: '):
            data_str = line_str[6:]
            if data_str == '[DONE]':
                break
            try:
                chunk = json.loads(data_str)
                if 'choices' in chunk and len(chunk['choices']) > 0:
                    delta = chunk['choices'][0].get('delta', {})
                    if 'reasoning_content' in delta:
                        sys.stdout.write(delta['reasoning_content'])
                        sys.stdout.flush()
                    if 'content' in delta:
                        sys.stdout.write(delta['content'])
                        sys.stdout.flush()
            except Exception as e:
                pass
print("\n[DONE]")
