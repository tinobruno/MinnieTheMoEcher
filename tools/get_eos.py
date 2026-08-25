import json

with open('tokenizer.json', 'r') as f:
    tokenizer = json.load(f)

vocab = tokenizer.get('model', {}).get('vocab', {})
for k, v in vocab.items():
    if 'end_of_sentence' in k or 'eos' in k or 'eot' in k or 'EOS' in k:
        print(f"Token: {k}, ID: {v}")

