import torch
import sys
from transformers import AutoTokenizer, AutoModelForCausalLM
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
messages = [
    {"role": "user", "content": "What is the capital of Germany?"},
    {"role": "assistant", "content": "The capital of Germany is Berlin.\n\nIt's the United Kingdom of which?"},
    {"role": "user", "content": "Where is it located?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("PROMPT:", repr(prompt))
input_ids = tokenizer.encode(prompt, return_tensors="pt")
print("LAST 5 TOKENS:", input_ids[0][-5:])
