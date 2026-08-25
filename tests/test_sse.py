import requests
import json
import sys

response = requests.post(
    "http://localhost:8080/v1/completions",
    json={
        "model": "deepseek-v4-flash",
        "prompt": "<｜begin▁of▁sentence｜><｜User｜>what is the commodore Amiga?<｜Assistant｜>",
        "max_tokens": 100,
        "temperature": 0.6,
        "repetition_penalty": 1.1,
        "stream": True
    },
    stream=True
)

for line in response.iter_lines():
    if line:
        print(repr(line.decode("utf-8")))
        sys.stdout.flush()
