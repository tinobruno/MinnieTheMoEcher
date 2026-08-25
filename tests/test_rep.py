import requests

prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the Commodore Amiga?<｜Assistant｜>"
req = {
    "prompt": prompt,
    "max_tokens": 128,
    "temperature": 0.6,
    "repetition_penalty": 1.0,  # NO PENALTY
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("No penalty:", repr(resp.json()["choices"][0]["text"]))

req["repetition_penalty"] = 1.1
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("Penalty 1.1:", repr(resp.json()["choices"][0]["text"]))
