import requests

prompt = "<｜begin▁of▁sentence｜><｜User｜>What is the capital of Italy?<｜Assistant｜>"
req = {
    "prompt": prompt,
    "max_tokens": 128,
    "temperature": 0.6,
    "repetition_penalty": 1.1,
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("Italy:", repr(resp.json()["choices"][0]["text"]))

prompt2 = "<｜begin▁of▁sentence｜><｜User｜>What is 2 + 2?<｜Assistant｜>"
req["prompt"] = prompt2
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("Math:", repr(resp.json()["choices"][0]["text"]))

