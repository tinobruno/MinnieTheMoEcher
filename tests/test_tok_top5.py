from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base", trust_remote_code=True)
for t in [2619, 2531, 16032, 666, 3167]:
    print(t, repr(tokenizer.decode([t])))
