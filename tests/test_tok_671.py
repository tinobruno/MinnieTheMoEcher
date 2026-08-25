from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
print(671, repr(tokenizer.decode([671])))
print(43, repr(tokenizer.decode([43])))
