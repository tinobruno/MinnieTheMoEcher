import requests, json

url = "http://127.0.0.1:8001/v1/chat/completions"

def query(prompt):
    payload = {
        "model": "deepseek-v4-flash",
        "messages": [{"role": "user", "content": prompt}],
        "max_tokens": 150,
        "temperature": 0.0,
        "stream": True
    }

    resp = requests.post(url, json=payload, stream=True)
    res = ""
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
                        res += delta
                except:
                    pass
    return res

prompts = [
    "What is the capital of Italy ?",
    "Who wrote Romeo and Juliet?",
    "Explain what a prime number is in two sentences."
]

for p in prompts:
    print(f"Q: {p}")
    print(f"A: {query(p)}")
    print("-" * 50)
