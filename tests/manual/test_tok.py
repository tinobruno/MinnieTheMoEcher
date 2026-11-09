import sys
sys.stdout.reconfigure(encoding='utf-8')
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained(r'F:\Moecher\models\deepseek_v41_nvfp4')
text = '<meta charset="UTF-8">'
tokens = tok.encode(text, add_special_tokens=False)
print('Tokens:', tokens)
for t in tokens:
    print(t, repr(tok.decode([t])))
