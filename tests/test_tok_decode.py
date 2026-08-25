from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
tokens = [25254, 492, 3167, 16214, 33]
for t in tokens:
    print(t, repr(tokenizer.decode([t])))
