import json

vocab = {}
with open("moecher_manifest_q2.json") as f:
    manifest = json.load(f)
    print(manifest["tokenizer"])

