import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"

prompts = [
    "What is the capital of Italy ?",
    "Where is Paris located?",
    "Who wrote Romeo and Juliet?",
    "What is 15 * 14?",
    "Name three common fruits.",
    "Explain gravity in one sentence.",
    "What is the chemical symbol for gold?"
]

for p in prompts:
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": p}],
        "max_tokens": 50,
        "temperature": 0.0,
        "stream": True
    }

    resp = requests.post(url, json=payload, stream=True)
    ans = ""
    for line in resp.iter_lines():
        if line:
            line_str = line.decode('utf-8')
            if line_str.startswith("data: "):
                data_str = line_str[6:]
                if data_str.strip() == "[DONE]":
                    break
                try:
                    data = json.loads(data_str)
                    delta = data['choices'][0]['delta'].get('content', '')
                    if delta:
                        ans += delta
                except:
                    pass
    print(f"Q: {p}")
    print(f"A: {ans.strip()}")
    print("-" * 50)
