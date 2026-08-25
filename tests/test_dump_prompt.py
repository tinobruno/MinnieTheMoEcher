from transformers import AutoTokenizer
import json
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')
prompt = tokenizer.apply_chat_template([{"role": "user", "content": "hi"}], tokenize=False, add_generation_prompt=True)
with open("prompt.bin", "wb") as f:
    f.write(prompt.encode('utf-8'))
