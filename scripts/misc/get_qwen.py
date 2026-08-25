from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('Qwen/Qwen2.5-Coder-7B-Instruct')
print(t.chat_template)
