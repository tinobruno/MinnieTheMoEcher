import json, sys
sys.stdout.reconfigure(encoding='utf-8')

with open(r'F:\Moecher\models\deepseek_v41_nvfp4\tokenizer.json', 'r', encoding='utf-8') as f:
    tok_data = json.load(f)

vocab = tok_data['model']['vocab']
inv_vocab = {v: k for k, v in vocab.items()}
for at in tok_data.get('added_tokens', []):
    inv_vocab[at['id']] = at['content']
    vocab[at['content']] = at['id']

messages = [
    {"role": "system", "content": "You are a helpful assistant that can write code, analyze data, and create complete applications using HTML, CSS, JavaScript, Python, and other programming languages. When asked to create or build something, always provide complete, functional, working code."},
    {"role": "user", "content": "create an HTML calculator"}
]

print("BOS:", vocab.get("<｜begin▁of▁sentence｜>"))
print("SYSTEM:", vocab.get("<｜System｜>"))
print("USER:", vocab.get("<｜User｜>"))
print("ASSISTANT:", vocab.get("<｜Assistant｜>"))
print("THINK_BEGIN:", vocab.get("<think>"))
print("THINK_END:", vocab.get("</think>"))
print("EOS:", vocab.get("<｜end▁of▁sentence｜>"))

# Let's see what multimodal.cpp currently does:
# 1. BOS (0)
# 2. SYSTEM (128799) + sys_content
# 3. USER (128803) + user_content
# 4. ASSISTANT (128804)
# 5. THINK_END (128822)
print("\nCurrent multimodal.cpp sequence ends with:")
print("... <｜Assistant｜> </think>")
