import json

with open("/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash/snapshots/60d8d70770c6776ff598c94bb586a859a38244f1/tokenizer.json", "r") as f:
    vocab = json.load(f)["model"]["vocab"]

inv_vocab = {v: k for k, v in vocab.items()}

tokens = [0, 128826, 3476, 591, 67, 18799, 1518, 14, 72, 16802]
for t in tokens:
    print(f"{t}: {repr(inv_vocab.get(t))}")
