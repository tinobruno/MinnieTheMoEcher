import requests, json

url = "http://localhost:8001/v1/chat/completions"
data = {
    "messages": [
        {"role": "user", "content": "What is 2+2? Think step by step."}
    ],
    "temperature": 0.0,
    "max_tokens": 512,
    "stream": True
}
response = requests.post(url, json=data, stream=True)
for line in response.iter_lines():
    if line:
        line_str = line.decode('utf-8')
        if line_str.startswith("data: ") and line_str != "data: [DONE]":
            chunk = json.loads(line_str[6:])
            delta = chunk["choices"][0]["delta"]
            if "reasoning_content" in delta:
                print(delta["reasoning_content"], end='', flush=True)
            if "content" in delta:
                print(delta["content"], end='', flush=True)
print()
