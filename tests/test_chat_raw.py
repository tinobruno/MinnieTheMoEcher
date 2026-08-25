import json, urllib.request

data = json.dumps({
    "model": "moecher",
    "prompt": "<｜begin▁of▁sentence｜><｜User｜>what is Amiga?<｜Assistant｜><think>\n",
    "stream": False,
    "max_tokens": 100
}).encode()

req = urllib.request.Request("http://localhost:8001/v1/completions", data=data, headers={"Content-Type": "application/json"})
resp = urllib.request.urlopen(req)
print(json.loads(resp.read())["choices"][0]["text"])
