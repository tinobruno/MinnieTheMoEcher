import requests
import json
resp = requests.post("http://localhost:8001/v1/chat/completions", json={
    "model": "moecher",
    "messages": [{"role": "user", "content": "Hello"}],
    "temperature": 0.0,
    "max_tokens": 100,
    "stream": True
}, stream=True)

reasoning = ""
content = ""
for line in resp.iter_lines():
    if line:
        decoded_line = line.decode('utf-8')
        if decoded_line.startswith("data: "):
            data_str = decoded_line[6:]
            if data_str == "[DONE]":
                break
            try:
                data = json.loads(data_str)
                delta = data["choices"][0].get("delta", {})
                if "reasoning_content" in delta:
                    reasoning += delta["reasoning_content"]
                if "content" in delta:
                    content += delta["content"]
            except json.JSONDecodeError:
                pass

print("REASONING REPR:", repr(reasoning))
print("CONTENT REPR:", repr(content))
