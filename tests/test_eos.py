import urllib.request
import json
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
messages = [
    {"role": "user", "content": "hello"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

req = urllib.request.Request(
    "http://localhost:8001/v1/completions",
    data=json.dumps({"prompt": prompt, "max_tokens": 10, "temperature": 0.6}).encode("utf-8"),
    headers={"Content-Type": "application/json"},
    method="POST",
)

with urllib.request.urlopen(req) as resp:
    res = json.loads(resp.read())
    text = res["choices"][0]["text"]
    print("Response repr:", repr(text))
    
