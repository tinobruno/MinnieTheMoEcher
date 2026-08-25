import requests
prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of Italy?<｜Assistant｜>"
req = {
    "prompt": prompt,
    "max_tokens": 128,
    "temperature": 0.0,
    "repetition_penalty": 1.0,
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("Response:", repr(resp.json()["choices"][0]["text"]))
