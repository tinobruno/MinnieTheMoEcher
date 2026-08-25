from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
messages = [{"role": "user", "content": "hello"}]
print(t.apply_chat_template(messages, tokenize=False))
