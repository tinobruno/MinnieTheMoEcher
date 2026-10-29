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

# Let's inspect what happens in multimodal.cpp:
# 1. BOS (0) -> '<｜begin▁of▁sentence｜>'
# 2. SYSTEM (128799) -> '<｜System｜>'
# 3. leading_sys = messages[0]['content']
# 4. USER (128803) -> '<｜User｜>'
# 5. user content
# 6. ASSISTANT (128804) -> '<｜Assistant｜>'
# 7. THINK_END (128822) -> '</think>'

# If we compare with master:
# On master:
# 1. BOS (0)
# 2. messages[0]['content'] (without <｜System｜>)
# 3. USER (128803)
# 4. user content
# 5. ASSISTANT (128804)
# 6. <think></think>\n

print("=== Case A: Current multimodal.cpp (with </think> only) ===")
print("<｜begin▁of▁sentence｜><｜System｜>" + messages[0]["content"] + "<｜User｜>" + messages[1]["content"] + "<｜Assistant｜></think>")

print("\n=== Case B: Master multimodal.cpp (with <think></think>\\n) ===")
print("<｜begin▁of▁sentence｜><｜System｜>" + messages[0]["content"] + "<｜User｜>" + messages[1]["content"] + "<｜Assistant｜><think></think>\n")

print("\n=== Case C: Master without System tag ===")
print("<｜begin▁of▁sentence｜>" + messages[0]["content"] + "<｜User｜>" + messages[1]["content"] + "<｜Assistant｜><think></think>\n")
