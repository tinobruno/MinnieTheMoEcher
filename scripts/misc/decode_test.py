from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
tokens = [0, 3476, 611, 3278, 304, 270, 2502, 6177, 1137, 46543]
for t in tokens:
    print(f"{t}: {repr(tokenizer.decode([t]))}")
