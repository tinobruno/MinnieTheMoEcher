import json
with open("moecher_manifest.json") as f: cfg = json.load(f)
with open(cfg["tokenizer"]["tokenizer_json"]) as f: tok = json.load(f)
for obj in tok.get("added_tokens", []):
    if obj["id"] in [128803, 128804]:
        print(f'{obj["id"]}: {obj["content"]}')
