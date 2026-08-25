import json

with open("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json") as f:
    t = json.load(f)

vocab = t["model"]["vocab"]
for token, tid in vocab.items():
    if "think" in token.lower():
        print(f"Token: {repr(token)} -> {tid}")
