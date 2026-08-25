import json

def get_bpe_decoding():
    v = json.load(open("moecher_manifest_q2.json"))
    tok_path = v["tokenizer"]["tokenizer_json"]
    tok = json.load(open(tok_path))
    vocab = tok["model"]["vocab"]
    inv_vocab = {v: k for k, v in vocab.items()}
    # Let's find tokens that start with < or l
    for i in range(100):
        if inv_vocab.get(i):
            print(f"{i}: {inv_vocab[i]}")

get_bpe_decoding()
