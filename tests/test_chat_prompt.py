from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
messages = [{"role": "user", "content": "what is the commodore Amiga?"}]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("Prompt:", repr(prompt))
print("Tokens:", tokenizer.encode(prompt))
