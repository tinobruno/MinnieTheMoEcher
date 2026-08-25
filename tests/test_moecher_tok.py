import requests
prompt = "<\uff5cAssistant\uff5c>"
req = {"prompt": prompt, "max_tokens": 1}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print(resp.json())
