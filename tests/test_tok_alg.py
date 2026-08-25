import json

# Load special tokens from json
with open('/home/tinobruno/.cache/huggingface/hub/models--deepseek-ai--DeepSeek-V4-Flash-0731/snapshots/7872f01b1d1fe23eabc4c98b48bffcef5a386062/tokenizer.json') as f:
    tok = json.load(f)

vocab = tok["model"]["vocab"]
special_tokens = []

if "added_tokens" in tok:
    for at in tok["added_tokens"]:
        special_tokens.append((at["content"], at["id"]))

# Sort special tokens by length descending (longest first)
special_tokens.sort(key=lambda x: len(x[0]), reverse=True)

text = "<｜begin▁of▁sentence｜><｜User｜>hello<｜Assistant｜>"
print("Text:", repr(text))

out = []
pos = 0
while pos < len(text):
    found = False
    for tok_str, tok_id in special_tokens:
        if text.startswith(tok_str, pos):
            out.append((tok_str, True, tok_id))
            pos += len(tok_str)
            found = True
            break
    if not found:
        if not out or out[-1][1]:
            out.append(["", False, -1])
        out[-1][0] += text[pos]
        pos += 1

print("Segments:", out)
