import json
with open('moecher_manifest.json') as f:
    cfg = json.load(f)
vocab = cfg['tokenizer']['vocab']
rev = {v: k for k, v in vocab.items()}
print("1536:", rev[1536])
print("279:", rev.get(279, "None"))
print("271:", rev.get(271, "None"))
