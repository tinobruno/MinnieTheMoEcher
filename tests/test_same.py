import json
import urllib.request
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
url = "http://127.0.0.1:8001/v1/completions"

def chat(messages, max_tokens=20, temp=0.0):
    prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    payload = json.dumps({
        "model": "deepseek-v4-flash",
        "prompt": prompt,
        "max_tokens": max_tokens,
        "temperature": temp,
        "stream": False
    }).encode("utf-8")

    req = urllib.request.Request(
        url, data=payload, headers={"Content-Type": "application/json"}, method="POST"
    )
    with urllib.request.urlopen(req) as resp:
        res = json.loads(resp.read().decode("utf-8"))
        return res["choices"][0]["text"]

messages = [
    {"role": "user", "content": "what is the capital of Italy?"}
]
print("Turn 1:", repr(chat(messages)))
print("Turn 2:", repr(chat(messages)))
print("Turn 3:", repr(chat(messages)))
