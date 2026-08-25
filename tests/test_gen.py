import requests, json, sys

url = "http://127.0.0.1:8001/v1/chat/completions"
payload = {
    "model": "deepseek-v4-flash",
    "messages": [
        {"role": "user", "content": "What is the capital of Italy ?"}
    ],
    "max_tokens": 150,
    "temperature": 0.0,
    "stream": True
}

resp = requests.post(url, json=payload, stream=True)
print("HTTP Status:", resp.status_code, flush=True)
print("=== GREEDY STREAM START ===", flush=True)
count = 0
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
                    sys.stdout.write(delta)
                    sys.stdout.flush()
                    count += 1
            except Exception as e:
                pass

print("\n=== GREEDY STREAM END ===", flush=True)
print(f"Total tokens emitted: {count}", flush=True)
