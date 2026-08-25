from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
print(2737, repr(tokenizer.decode([2737])))
