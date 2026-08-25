import urllib.request
import json

req = urllib.request.Request("http://localhost:8001/v1/chat/completions",
    data=json.dumps({
        "messages": [{"role": "user", "content": "hello"}],
        "max_tokens": 100,
        "stream": False
    }).encode('utf-8'),
    headers={'Content-Type': 'application/json'}
)

with urllib.request.urlopen(req) as response:
    print(json.dumps(json.loads(response.read().decode('utf-8')), indent=2))
