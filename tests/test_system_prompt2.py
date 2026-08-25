import json
import requests
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3")
messages = [
    {"role": "system", "content": "You are a helpful AI assistant."},
    {"role": "user", "content": "what is the commodore Amiga?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

req = {
    "prompt": prompt,
    "max_tokens": 128,
    "temperature": 0.6,
    "repetition_penalty": 1.1,
    "stream": False
}
resp = requests.post("http://localhost:8001/v1/completions", json=req)
print(resp.json()["choices"][0]["text"])
