import json

with open("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json") as f:
    vocab = json.load(f)["model"]["vocab"]

inv_vocab = {v: k for k, v in vocab.items()}
tokens = [0, 128803, 3085, 344, 270, 6102, 294, 14251, 33, 128804]
for t in tokens:
    print(f"{t}: {inv_vocab.get(t, '???')}")
