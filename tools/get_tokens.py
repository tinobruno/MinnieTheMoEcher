import json
with open("moecher_manifest.json") as f: cfg = json.load(f)
with open(cfg["tokenizer"]["tokenizer_json"]) as f: tok = json.load(f)
vocab = tok["model"]["vocab"]
rev = {v: k for k, v in vocab.items()}
for t in [0, 128803, 23166, 128804]:
    print(f"{t}: {rev.get(t, 'None')}")
