import requests

prompt = "<｜begin▁of▁sentence｜><｜User｜>what is the commodore Amiga?<｜Assistant｜>"
req = {
    "prompt": prompt,
    "max_tokens": 128,
    "temperature": 0.6,
    "repetition_penalty": 1.1,
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("Response:", repr(resp.json()["choices"][0]["text"]))
