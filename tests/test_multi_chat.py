from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained('deepseek-ai/DeepSeek-V3')
messages = [
    {"role": "user", "content": "hello"},
    {"role": "assistant", "content": "hi"},
    {"role": "user", "content": "what is your name?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print(repr(prompt))
print("Encode via HF:", tokenizer.encode(prompt))
