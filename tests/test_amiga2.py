import requests
import json
prompt = "<\xef\xbd\x9cbegin\xe2\x96\x81of\xe2\x96\x81sentence\xef\xbd\x9c><\xef\xbd\x9cUser\xef\xbd\x9c>what is the commodore Amiga?<\xef\xbd\x9cAssistant\xef\xbd\x9c>"
req = {
    "prompt": prompt,
    "max_tokens": 100,
    "temperature": 0.6,
    "repetition_penalty": 1.1,
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/chat/completions", json=req)
print("Amiga:", resp.text)
