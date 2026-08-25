from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')
ids = tokenizer.encode("What is the Commodore Amiga?")
print(ids)
for i in ids: print(repr(tokenizer.decode([i])))
