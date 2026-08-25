import requests
prompt = "What is the Commodore Amiga?"
req = {"prompt": prompt, "max_tokens": 1, "temperature": 0.0, "stream": False}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print(resp.json())
