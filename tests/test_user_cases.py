import json
import urllib.request
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
url = "http://127.0.0.1:8001/v1/chat/completions"

def chat(messages, max_tokens=512, temp=0.0):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    payload = json.dumps({
        "model": "deepseek-v4-flash",
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["message"]["content"]

hist = [{"role": "user", "content": "What is the capital of Germany?"}]
r1 = chat(hist)
print("Turn 1 Q: What is the capital of Germany?")
print("Turn 1 A:", repr(r1))


hist.append({"role": "assistant", "content": r1})
hist.append({"role": "user", "content": "Where is it located?"})
r2 = chat(hist)
print("Turn 2 Q: Where is it located?")
print("Turn 2 A:", repr(r2))
