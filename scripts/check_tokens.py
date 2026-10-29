import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'F:\Moecher\models\deepseek_v41_nvfp4\tokenizer.json', 'r', encoding='utf-8') as f:
    data = json.load(f)

vocab = data['model']['vocab']
print('Vocab size:', len(vocab))

tokens_to_test = [
    '<｜begin of sentence｜>', '<｜end of sentence｜>',
    '<｜begin▁of▁sentence｜>', '<｜end▁of▁sentence｜>',
    '<｜System｜>', '<｜System▁｜>',
    '<｜User｜>', '<｜User▁｜>',
    '<｜Assistant｜>', '<｜Assistant▁｜>',
    '<think>', '</think>'
]
for t in tokens_to_test:
    tid = vocab.get(t, None)
    print(f'{repr(t)}: {tid}')

added = data.get('added_tokens', [])
print(f'Total added tokens: {len(added)}')
for at in added:
    c = at['content']
    if any(k in c for k in ['System', 'User', 'Assistant', 'think', 'sentence', 'tool', 'begin', 'end']):
        print(f'{repr(c)}: id={at["id"]} special={at.get("special")}')
