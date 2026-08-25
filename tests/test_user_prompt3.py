import json
import requests

req = {
    "prompt": "<｜begin▁of▁sentence｜><｜User｜>what is the commodore Amiga?<｜Assistant｜>",
    "max_tokens": 128,
    "temperature": 0.0,
    "repetition_penalty": 1.1,
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print(resp.json()["choices"][0]["text"])
