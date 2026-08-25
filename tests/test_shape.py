import json

with open("moecher_manifest.json") as f:
    d = json.load(f)

print(d["model_config"])
