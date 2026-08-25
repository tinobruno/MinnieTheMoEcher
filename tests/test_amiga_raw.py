import requests
req = {
    "prompt": "<\uff5cbegin\u2581of\u2581sentence\uff5c><\uff5cUser\uff5c>what is the commodore Amiga?<\uff5cAssistant\uff5c>",
    "max_tokens": 512,
    "temperature": 0.6
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print("Status:", resp.status_code)
print("Text:", resp.text)
