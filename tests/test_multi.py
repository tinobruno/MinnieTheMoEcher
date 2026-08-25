import urllib.request
import urllib.error
import json
import time

def wait_for_server():
    for _ in range(60):
        try:
            # use a valid prompt so if it succeeds, it just quickly generates one token 
            # wait, better to just use urllib.request.urlopen with a GET and expect 405
            req = urllib.request.Request("http://localhost:8001/v1/chat/completions", method="GET")
            urllib.request.urlopen(req, timeout=1)
            return
        except urllib.error.HTTPError as e:
            if e.code in [404, 405]:
                return
        except Exception:
            pass
        time.sleep(1)

def chat(messages, rep_penalty):
    payload = json.dumps({
        "model": "deepseek-v4-flash-0731",
        "messages": messages,
        "max_tokens": 100,
        "temperature": 0.0,
        "repetition_penalty": rep_penalty
    }).encode("utf-8")
    req = urllib.request.Request("http://localhost:8001/v1/chat/completions", data=payload, headers={"Content-Type": "application/json"}, method="POST")
    with urllib.request.urlopen(req) as resp:
        return json.loads(resp.read().decode())['choices'][0]['message']['content']

if __name__ == "__main__":
    wait_for_server()
    messages = [{"role": "user", "content": "What is the capital of Italy?"}]
    reply1 = chat(messages, 1.0)
    print("Turn 1:", reply1)
    
    messages.append({"role": "assistant", "content": reply1})
    messages.append({"role": "user", "content": "And the capital of Belgium?"})
    reply2 = chat(messages, 1.0)
    print("Turn 2:", reply2)
