from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V3-Base")
print("</think>:", t.encode("</think>", add_special_tokens=False))
print("<｜think｜>:", t.encode("<｜think｜>", add_special_tokens=False))
