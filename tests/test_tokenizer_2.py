from transformers import AutoTokenizer
tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3", trust_remote_code=True)
print("</think>:", tokenizer.encode("</think>"))
print("<｜think｜>:", tokenizer.encode("<｜think｜>"))
print("<｜/think｜>:", tokenizer.encode("<｜/think｜>"))
