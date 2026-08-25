from transformers import AutoTokenizer
model_path = "deepseek-ai/DeepSeek-V4-Flash-0731"
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
messages = [
    {"role": "user", "content": "What is the capital of Italy?"},
    {"role": "assistant", "content": "The capital of Italy is Rome, which is also the largest city of the country. It is located in the central part of the country, on the banks of the River Tiber. As the capital of Italy, Rome is not only the political center of the country, but also a famous historical and cultural city in the world."},
    {"role": "user", "content": "And the capital of Belgium?"}
]
prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
print("HF Prompt:")
print(repr(prompt))
ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True)
print("HF Token IDs:", ids)
