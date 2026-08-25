import json

with open("moecher_manifest.json") as f:
    d = json.load(f)

for k, v in d.items():
    if "layers.0.attn." in k and "weight" in k:
        print(f"{k}: {v['shape']}")
